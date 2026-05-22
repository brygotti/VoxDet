#!/usr/bin/env python3
"""
Full-pipeline token / query / FLOPs breakdown for
configs/foveated-peri-maxpool-semantickitti-cam.py

Run from the VoxDet repo root:
    python tools/full_pipeline_analysis.py
"""

import math
import torch

# ─── Config params ────────────────────────────────────────────────────────────
IMG_H, IMG_W       = 384, 1280          # input image resolution
DOWNSAMPLE         = 8                  # backbone+neck spatial stride
FEAT_H, FEAT_W     = IMG_H // DOWNSAMPLE, IMG_W // DOWNSAMPLE  # 48 × 160
DEPTH_START, DEPTH_END, DEPTH_STEP = 2.0, 58.0, 0.5
D                  = round((DEPTH_END - DEPTH_START) / DEPTH_STEP)  # 112 depth bins
NUM_CAMS           = 1

# Volume / query grid (after lss_downsample=[2,2,2])
VOLUME_H, VOLUME_W, VOLUME_Z = 128, 128, 16
N_VOXELS           = VOLUME_H * VOLUME_W * VOLUME_Z  # 262 144

# Foveated zones
FOVEAL_RADIUS      = 0.2
MID_RADIUS         = 0.4
MID_STRIDE         = 3
PERI_STRIDE        = 5
FIXATION           = [0.25, 0.5, 0.5]
ALIGN              = 4

# FoveatedLocalAggregator 3-D backbone / FPN
NUM_LAYER_3D       = [2, 2, 2]
NUM_CHANNELS_3D    = [128, 128, 128]
STRIDE_3D          = [1, 2, 2]
C                  = 128               # feature channels throughout

# Cross-attention (VoxFormerHead)
EMBED_DIM          = 128
NUM_LAYERS_XA      = 3
NUM_POINTS_XA      = 8
NUM_LEVELS_XA      = 1
FF_DIM             = 1024              # feedforward_channels in FFN config

# OccHead
OCC_SIZE           = [256, 256, 32]   # final output resolution
NUM_CLASS          = 20

# ResNet-50 reference FLOPs (224×224) from published benchmarks
RESNET50_REF_GFLOPS = 4.1             # GFLOPs @ 224×224

SEP  = "=" * 72
HSEP = "-" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aligned_bound(center, radius, N, align):
    lo_f = (center - radius) * N
    hi_f = (center + radius) * N
    lo = max(0, math.floor(lo_f))
    hi = min(N, math.ceil(hi_f))
    size = hi - lo
    aligned = math.ceil(size / align) * align
    extra = aligned - size
    lo = max(0, lo - extra // 2)
    hi = lo + aligned
    if hi > N:
        hi = N
        lo = hi - aligned
    return max(0, lo), hi


def count_tokens():
    """Exact token counts per foveated zone (mirrors VoxFormerHead._pool_zone)."""
    fx, fy, fz = FIXATION
    xs = (torch.arange(VOLUME_H, dtype=torch.float32) + 0.5) / VOLUME_H
    ys = (torch.arange(VOLUME_W, dtype=torch.float32) + 0.5) / VOLUME_W
    zs = (torch.arange(VOLUME_Z, dtype=torch.float32) + 0.5) / VOLUME_Z

    dx = (xs - fx).abs()
    dy = (ys - fy).abs()
    dz = (zs - fz).abs()
    D3 = torch.max(dx[:, None, None], torch.max(dy[None, :, None], dz[None, None, :]))

    foveal_mask = D3 <= FOVEAL_RADIUS
    mid_mask    = (D3 > FOVEAL_RADIUS) & (D3 <= MID_RADIUS)
    peri_mask   = D3 > MID_RADIUS

    def unique_blocks(mask, stride):
        idx = mask.nonzero()
        if idx.numel() == 0:
            return 0
        bx = idx[:, 0] // stride
        by = idx[:, 1] // stride
        bz = idx[:, 2] // stride
        nb_y = (VOLUME_W + stride - 1) // stride
        nb_z = (VOLUME_Z + stride - 1) // stride
        block_ids = bx * (nb_y * nb_z) + by * nb_z + bz
        return int(block_ids.unique().numel())

    return dict(
        baseline       = N_VOXELS,
        foveal_voxels  = int(foveal_mask.sum()),
        mid_voxels     = int(mid_mask.sum()),
        peri_voxels    = int(peri_mask.sum()),
        foveal_tokens  = int(foveal_mask.sum()),
        mid_tokens     = unique_blocks(mid_mask,  MID_STRIDE),
        peri_tokens    = unique_blocks(peri_mask, PERI_STRIDE),
    )


def analytical_flops_3d(volume_shape, num_layer, num_channels, strides,
                         c_in=128, fpn_out=128):
    """MACs for CustomResNet3D + GeneralizedLSSFPN (conv3d version)."""
    H, W, Z = volume_shape
    V = H * W * Z
    K3 = 27  # 3×3×3

    backbone_macs = 0
    stage_volumes = []
    curr_c, curr_v = c_in, V

    for n_blk, out_c, s in zip(num_layer, num_channels, strides):
        out_v = curr_v // (s ** 3)
        b1  = (curr_c * K3 * out_c) * out_v
        b1 += (out_c  * K3 * out_c) * out_v
        if s != 1 or curr_c != out_c:
            b1 += (curr_c * K3 * out_c) * out_v
        b_rest = 2 * (out_c * K3 * out_c) * out_v
        backbone_macs += b1 + (n_blk - 1) * b_rest
        stage_volumes.append(out_v)
        curr_c, curr_v = out_c, out_v

    fpn_macs = 0
    n_lvl = len(stage_volumes)
    for lvl in range(n_lvl - 1, -1, -1):
        vol  = stage_volumes[lvl]
        in_c = num_channels[lvl] if lvl == n_lvl - 1 else num_channels[lvl] + fpn_out
        fpn_macs += in_c    * 1  * fpn_out * vol
        fpn_macs += fpn_out * K3 * fpn_out * vol

    total = backbone_macs + fpn_macs
    return dict(backbone_g=backbone_macs / 1e9,
                fpn_g=fpn_macs / 1e9,
                total_g=total / 1e9)


# ─────────────────────────────────────────────────────────────────────────────
# Stage FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def stage1_resnet50():
    """ResNet-50 scaled linearly from 224×224 reference."""
    scale = (IMG_H * IMG_W) / (224 * 224)
    return RESNET50_REF_GFLOPS * scale


def stage2_secondfpn():
    """
    SECONDFPN: 4 deconv/conv blocks, each to (FEAT_H, FEAT_W).
    Uses ConvTranspose2d(C_in, C_out, kernel=stride, stride=stride)
    except level-0 which downsamples via Conv2d(stride=2, kernel=3).
    """
    V2d = FEAT_H * FEAT_W  # 7 680
    in_ch  = [256, 512, 1024, 2048]
    out_ch = [128, 128, 128,  128]
    up_str = [0.5, 1,   2,    4]

    total = 0
    for c_in, c_out, us in zip(in_ch, out_ch, up_str):
        if us < 1:           # downsample: Conv2d(3×3, stride=2)
            total += c_in * 9 * c_out * V2d
        elif us == 1:        # lateral: Conv2d(1×1)
            total += c_in * 1 * c_out * V2d
        else:                # upsample: ConvTranspose2d(k=us, stride=us)
            k = int(us)
            total += c_in * (k * k) * c_out * V2d
    return total / 1e9


def stage3_depth_net():
    """
    Rough closed-form for GeometryDepth_Net:
    - DepthNet: a few Conv2d on (512, 48, 160) → (128+D, 48, 160)
    - StereoVolumeEncoder: small UNet on (D, 48, 160)
    Estimated as 3 conv layers with representative channel counts.
    """
    V2d = FEAT_H * FEAT_W  # 7 680
    # DepthNet: 3 × Conv2d(512, 512, 3) + 1 × Conv2d(512, 128+D, 1)
    depth_net_macs = (3 * 512 * 9 * 512 + 512 * 1 * (C + D)) * V2d
    # StereoVolumeEncoder: Conv + UNet(D) + Conv: approximated as 3 layers on D-channel feature
    stereo_macs = 3 * D * 9 * D * V2d
    return (depth_net_macs + stereo_macs) / 1e9


def stage4_lss():
    """
    LSS View Transformer: frustum sampling + voxel pooling (scatter).
    No multiply-adds on features — index-based scatter, FLOPs ≈ 0.
    Returns frustum point count for reporting.
    """
    frustum_pts = D * FEAT_H * FEAT_W  # 112 × 48 × 160 = 860 160
    return 0.0, frustum_pts


def stage5_foveated_aggregator():
    """FoveatedLocalAggregator: foveal crop + global stream."""
    fx, fy, fz = FIXATION
    x0, x1 = _aligned_bound(fx, FOVEAL_RADIUS, VOLUME_H, ALIGN)
    y0, y1 = _aligned_bound(fy, FOVEAL_RADIUS, VOLUME_W, ALIGN)
    z0, z1 = _aligned_bound(fz, FOVEAL_RADIUS, VOLUME_Z, ALIGN)

    fov_shape  = (x1 - x0, y1 - y0, z1 - z0)
    glob_shape = (VOLUME_H // 2, VOLUME_W // 2, VOLUME_Z // 2)
    full_shape = (VOLUME_H, VOLUME_W, VOLUME_Z)

    base = analytical_flops_3d(full_shape, NUM_LAYER_3D, NUM_CHANNELS_3D, STRIDE_3D)
    fov  = analytical_flops_3d(fov_shape,  NUM_LAYER_3D, NUM_CHANNELS_3D, STRIDE_3D)
    glob = analytical_flops_3d(glob_shape, NUM_LAYER_3D, NUM_CHANNELS_3D, STRIDE_3D)

    return dict(
        fov_shape=fov_shape,  glob_shape=glob_shape,
        base=base,  fov=fov,  glob=glob,
        fov_volume=fov_shape[0]*fov_shape[1]*fov_shape[2],
        glob_volume=glob_shape[0]*glob_shape[1]*glob_shape[2],
        foveated_total_g=fov['total_g'] + glob['total_g'],
        baseline_total_g=base['total_g'],
    )


def stage6_voxformer_head(tok):
    """
    VoxFormerHead: mlp_lss on all voxels + deformable cross-attention on active tokens.
    Per-query-per-layer cost (MACs):
      q-proj (D²) + offset (D×P×3) + attn-weight (D×P) + weighted-sum (P×D) + out-proj (D²) + FFN (2×D×FF)
    Plus image-side (value sampling, amortised): negligible for large Q.
    """
    foveal_t = tok['foveal_tokens']
    mid_t    = tok['mid_tokens']
    peri_t   = tok['peri_tokens']
    Q_fov    = foveal_t + mid_t + peri_t   # active query tokens (foveated)
    Q_dense  = N_VOXELS                     # all voxels (dense baseline)

    # mlp_lss: 2 × Linear(D, D) applied to ALL voxels
    mlp_lss_macs_per_tok = 2 * EMBED_DIM * EMBED_DIM  # 32 768
    mlp_lss_g = N_VOXELS * mlp_lss_macs_per_tok / 1e9

    # Deformable cross-attention per query per layer
    per_q = (EMBED_DIM * EMBED_DIM           # q-proj
           + EMBED_DIM * NUM_POINTS_XA * 3   # sampling offset (3-D)
           + EMBED_DIM * NUM_POINTS_XA        # attention weights
           + NUM_POINTS_XA * EMBED_DIM        # weighted sum
           + EMBED_DIM * EMBED_DIM            # output proj
           + 2 * EMBED_DIM * FF_DIM)          # FFN (2 linear layers)

    xa_fov_g   = NUM_LAYERS_XA * Q_fov   * per_q / 1e9
    xa_dense_g = NUM_LAYERS_XA * Q_dense * per_q / 1e9

    # mlp_prior (empty voxels only; assume ~50 % proposal fill)
    empty_vox   = N_VOXELS // 2
    prior_per_q = 2 * (EMBED_DIM * (EMBED_DIM // 2) + (EMBED_DIM // 2) * EMBED_DIM)
    mlp_prior_g = empty_vox * prior_per_q / 1e9

    return dict(
        Q_fov=Q_fov, Q_dense=Q_dense, per_q_macs=per_q,
        mlp_lss_g=mlp_lss_g, mlp_prior_g=mlp_prior_g,
        xa_fov_g=xa_fov_g, xa_dense_g=xa_dense_g,
        total_fov_g=mlp_lss_g + xa_fov_g + mlp_prior_g,
        total_dense_g=mlp_lss_g + xa_dense_g + mlp_prior_g,
    )


def stage7_occ_head():
    """
    OccHead: Conv3d(128→64, k=3) + Conv3d(64→20, k=1) on 128×128×16,
    then trilinear upsample to occ_size.
    """
    V = N_VOXELS  # 262 144
    conv1 = C * 27 * (C // 2) * V           # 128→64, 3×3×3
    conv2 = (C // 2) * 1 * NUM_CLASS * V    # 64→20, 1×1×1
    return (conv1 + conv2) / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    tok = count_tokens()
    tok['foveated_total'] = tok['foveal_tokens'] + tok['mid_tokens'] + tok['peri_tokens']

    r50_g    = stage1_resnet50()
    fpn_g    = stage2_secondfpn()
    depth_g  = stage3_depth_net()
    lss_g, frustum_pts = stage4_lss()
    agg      = stage5_foveated_aggregator()
    xa       = stage6_voxformer_head(tok)
    occ_g    = stage7_occ_head()

    total_fov_g   = r50_g + fpn_g + depth_g + lss_g + agg['foveated_total_g'] + xa['total_fov_g']   + occ_g
    total_dense_g = r50_g + fpn_g + depth_g + lss_g + agg['baseline_total_g']  + xa['total_dense_g'] + occ_g

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FULL PIPELINE: Tokens · Queries · FLOPs")
    print("  Config: foveated-peri-maxpool-semantickitti-cam.py")
    print(SEP)
    print(f"  Image: {IMG_H}×{IMG_W}  |  Feature map: {FEAT_H}×{FEAT_W}  |"
          f"  D={D} depth bins  |  1 camera")
    print(f"  Query volume: {VOLUME_H}×{VOLUME_W}×{VOLUME_Z} = {N_VOXELS:,} voxels")
    print(f"  Fixation: {FIXATION}  |  radii: foveal≤{FOVEAL_RADIUS}, mid≤{MID_RADIUS},"
          f" peripheral (stride {PERI_STRIDE})")
    print(SEP)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 1 — Image Backbone (ResNet-50)")
    print(f"  {HSEP}")
    print(f"  Input  : ({IMG_H}, {IMG_W})")
    print(f"  Output : 4 feature maps, channels [256,512,1024,2048]")
    print(f"  Tokens : dense CNN — no discrete tokens")
    print(f"  FLOPs  : {r50_g:.1f} G  "
          f"(ResNet-50 ref {RESNET50_REF_GFLOPS} G × {IMG_H*IMG_W/(224*224):.2f}× scale)")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 2 — Image Neck (SECONDFPN)")
    print(f"  {HSEP}")
    print(f"  Input  : [256,512,1024,2048] at 4 spatial scales → [128,128,128,128]")
    print(f"  Output : 4 × 128 ch all at ({FEAT_H},{FEAT_W})")
    print(f"  Tokens : dense CNN — no discrete tokens")
    print(f"  FLOPs  : {fpn_g:.1f} G")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 3 — Depth Net (GeometryDepth_Net)")
    print(f"  {HSEP}")
    print(f"  Input  : ({C*4},{FEAT_H},{FEAT_W})  |  D={D} depth bins")
    print(f"  Output : context ({C},{FEAT_H},{FEAT_W}) + depth ({D},{FEAT_H},{FEAT_W})")
    print(f"  Tokens : dense CNN — no discrete tokens")
    print(f"  FLOPs  : {depth_g:.1f} G  (DepthNet + StereoVolumeEncoder, estimated)")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 4 — LSS View Transformer")
    print(f"  {HSEP}")
    print(f"  Frustum: {D}×{FEAT_H}×{FEAT_W} = {frustum_pts:,} 3-D sample points")
    print(f"  Output : ({C},{VOLUME_H},{VOLUME_W},{VOLUME_Z}) = {N_VOXELS:,} voxel queries")
    print(f"  FLOPs  : ~0  (index-based scatter / voxel pooling, no MACs)")

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 5 — occ_encoder_neck (FoveatedLocalAggregator)")
    print(f"  {HSEP}")
    fs = agg['fov_shape']; gs = agg['glob_shape']
    print(f"  Foveal crop    : x[{_aligned_bound(FIXATION[0],FOVEAL_RADIUS,VOLUME_H,ALIGN)[0]}"
          f":{_aligned_bound(FIXATION[0],FOVEAL_RADIUS,VOLUME_H,ALIGN)[1]}] "
          f"y[{_aligned_bound(FIXATION[1],FOVEAL_RADIUS,VOLUME_W,ALIGN)[0]}"
          f":{_aligned_bound(FIXATION[1],FOVEAL_RADIUS,VOLUME_W,ALIGN)[1]}] "
          f"z[{_aligned_bound(FIXATION[2],FOVEAL_RADIUS,VOLUME_Z,ALIGN)[0]}"
          f":{_aligned_bound(FIXATION[2],FOVEAL_RADIUS,VOLUME_Z,ALIGN)[1]}]"
          f" → {fs}  V={agg['fov_volume']:,}")
    print(f"  Global (2× pool): {gs}  V={agg['glob_volume']:,}")
    print()
    print(f"  {'Stream':<38} {'Backbone':>9}  {'FPN':>7}  {'Total':>7}")
    print(f"  {HSEP}")
    b = agg['base']; f = agg['fov']; g = agg['glob']
    print(f"  {'Dense baseline (128×128×16)':<38} {b['backbone_g']:>8.1f}G  {b['fpn_g']:>6.1f}G  {b['total_g']:>6.1f}G")
    print(f"  {'Foveal stream '+str(fs):<38} {f['backbone_g']:>8.1f}G  {f['fpn_g']:>6.1f}G  {f['total_g']:>6.1f}G")
    print(f"  {'Global stream '+str(gs):<38} {g['backbone_g']:>8.1f}G  {g['fpn_g']:>6.1f}G  {g['total_g']:>6.1f}G")
    print(f"  {HSEP}")
    print(f"  {'Foveated total':<38} {'':>9}  {'':>7}  {agg['foveated_total_g']:>6.1f}G")
    print(f"  {'Reduction vs dense':<38} {'':>9}  {'':>7}  {agg['baseline_total_g']/agg['foveated_total_g']:>6.1f}×")

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 6 — VoxFormerHead (Deformable Cross-Attention)")
    print(f"  {HSEP}")
    print(f"  Query volume   : {N_VOXELS:,} voxels  ({VOLUME_H}×{VOLUME_W}×{VOLUME_Z})")
    print(f"  Image tokens   : {FEAT_H}×{FEAT_W} = {FEAT_H*FEAT_W:,} (context feature map)")
    print()
    print(f"  Foveated zone pooling:")
    print(f"  {'Zone':<26} {'Voxels':>10}  {'Tokens':>10}  {'Compression':>12}")
    print(f"  {HSEP}")
    print(f"  {'Baseline (dense)':<26} {tok['baseline']:>10,}  {tok['baseline']:>10,}  {'1.0×':>12}")
    print(f"  {'Foveal (stride 1)':<26} {tok['foveal_voxels']:>10,}  {tok['foveal_tokens']:>10,}  {'1.0×':>12}")
    print(f"  {'Mid (stride '+str(MID_STRIDE)+')':<26} {tok['mid_voxels']:>10,}  {tok['mid_tokens']:>10,}"
          f"  {tok['mid_voxels']/max(1,tok['mid_tokens']):>11.1f}×")
    print(f"  {'Peripheral (stride '+str(PERI_STRIDE)+')':<26} {tok['peri_voxels']:>10,}  {tok['peri_tokens']:>10,}"
          f"  {tok['peri_voxels']/max(1,tok['peri_tokens']):>11.1f}×")
    print(f"  {HSEP}")
    print(f"  {'Foveated total':<26} {'':>10}   {tok['foveated_total']:>10,}  "
          f"{tok['baseline']/tok['foveated_total']:>11.1f}×")
    print()
    print(f"  Cross-attention: {NUM_LAYERS_XA} layers, D={EMBED_DIM}, P={NUM_POINTS_XA} points/query, FF={FF_DIM}")
    print(f"  Per-query-per-layer: {xa['per_q_macs']:,} MACs")
    print()
    print(f"  {'Sub-stage':<36} {'Foveated':>10}  {'Dense':>10}")
    print(f"  {HSEP}")
    print(f"  {'mlp_lss (all voxels, 2×Linear)':<36} {xa['mlp_lss_g']:>9.2f}G  {xa['mlp_lss_g']:>9.2f}G")
    print(f"  {'Cross-attention ('+str(NUM_LAYERS_XA)+' layers)':<36} {xa['xa_fov_g']:>9.2f}G  {xa['xa_dense_g']:>9.2f}G")
    print(f"  {'mlp_prior (empty voxels ~50%)':<36} {xa['mlp_prior_g']:>9.2f}G  {xa['mlp_prior_g']:>9.2f}G")
    print(f"  {HSEP}")
    print(f"  {'Stage 6 total':<36} {xa['total_fov_g']:>9.2f}G  {xa['total_dense_g']:>9.2f}G")
    print(f"  {'Active queries':<36} {xa['Q_fov']:>10,}  {xa['Q_dense']:>10,}")

    # ── Stage 7 ───────────────────────────────────────────────────────────────
    print(f"\n  STAGE 7 — OccHead (pts_bbox_head)")
    print(f"  {HSEP}")
    print(f"  Conv3d({C}→{C//2}, k=3×3×3) on {VOLUME_H}×{VOLUME_W}×{VOLUME_Z}")
    conv1_g = (C * 27 * (C // 2) * N_VOXELS) / 1e9
    conv2_g = ((C // 2) * 1 * NUM_CLASS * N_VOXELS) / 1e9
    print(f"  Conv3d({C//2}→{NUM_CLASS}, k=1×1×1) on {VOLUME_H}×{VOLUME_W}×{VOLUME_Z}")
    print(f"  Trilinear upsample: {VOLUME_H}×{VOLUME_W}×{VOLUME_Z} → {OCC_SIZE}")
    print(f"  Tokens : dense CNN — no discrete tokens")
    print(f"  FLOPs  : {conv1_g:.2f}G (conv1) + {conv2_g:.2f}G (conv2) = {occ_g:.2f}G total")

    # ── Grand Total ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GRAND TOTAL SUMMARY")
    print(SEP)
    header = f"  {'Stage':<42} {'Tokens/Queries':>16}  {'Foveated':>9}  {'Dense':>9}"
    print(header)
    print(f"  {HSEP}")

    rows = [
        ("1. ResNet-50 backbone",      "—  (dense CNN)",           r50_g,                   r50_g),
        ("2. SECONDFPN neck",           "—  (dense CNN)",           fpn_g,                   fpn_g),
        ("3. GeometryDepth_Net",        "—  (dense CNN)",           depth_g,                 depth_g),
        ("4. LSS View Transformer",     f"→ {N_VOXELS:,} voxels",  lss_g,                   lss_g),
        ("5. FoveatedLocalAggregator",  f"{N_VOXELS:,} voxels in", agg['foveated_total_g'], agg['baseline_total_g']),
        ("6. VoxFormerHead (XA)",       f"Q={xa['Q_fov']:,} / {xa['Q_dense']:,}", xa['total_fov_g'], xa['total_dense_g']),
        ("7. OccHead",                  "—  (dense CNN)",           occ_g,                   occ_g),
    ]
    for name, tok_str, fg, dg in rows:
        print(f"  {name:<42} {tok_str:>16}  {fg:>8.1f}G  {dg:>8.1f}G")

    print(f"  {HSEP}")
    print(f"  {'TOTAL':<42} {'':>16}  {total_fov_g:>8.1f}G  {total_dense_g:>8.1f}G")
    print(f"  {'Overall FLOPs reduction (foveated vs dense)':<42} {'':>16}  {total_dense_g/total_fov_g:>8.1f}×")
    print()
    print(f"  Key savings:")
    print(f"    Cross-attention queries : {xa['Q_fov']:,} vs {xa['Q_dense']:,}  →  {xa['Q_dense']/xa['Q_fov']:.1f}× fewer queries")
    print(f"    3D encoder (FovAgg)     : {agg['foveated_total_g']:.1f}G vs {agg['baseline_total_g']:.1f}G  →  {agg['baseline_total_g']/agg['foveated_total_g']:.1f}× fewer FLOPs")
    print(f"    Total pipeline          : {total_fov_g:.1f}G vs {total_dense_g:.1f}G  →  {total_dense_g/total_fov_g:.1f}×")
    print(f"\n{SEP}\n")


if __name__ == '__main__':
    main()
