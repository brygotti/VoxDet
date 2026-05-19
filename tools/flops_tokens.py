#!/usr/bin/env python3
"""
Verify token counts and FLOPs for FoveatedLocalAggregator vs dense baseline.

Run from the VoxDet repo root with the project environment activated:
    python scripts/flops_tokens.py

Section 1 — Transformer token counts per zone (exact, matches _pool_zone logic).
Section 2 — Analytical backbone+FPN FLOPs (closed-form, no forward pass needed).
Section 3 — Live FLOPs via fvcore or thop (optional, requires model import).
"""

import math
import sys
import os

import torch
import torch.nn as nn

# ── Configuration — must match configs/foveated-backbone-dev-semantickitti-cam.py ──
VOLUME_H = 128
VOLUME_W = 128
VOLUME_Z = 16
C = 128

FOVEAL_RADIUS      = 0.2
MID_RADIUS         = 0.4
MID_STRIDE         = 3
PERIPHERAL_STRIDE  = 5
FIXATION           = [0.25, 0.5, 0.5]
ALIGN              = 4       # alignment for foveal crop in FoveatedLocalAggregator

NUM_LAYER    = [2, 2, 2]
NUM_CHANNELS = [128, 128, 128]
STRIDE       = [1, 2, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Transformer Token Counts
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens(H, W, Z, fixation, foveal_r, mid_r, mid_stride, peri_stride):
    """Exact token counts using the same block-grouping logic as _pool_zone."""
    fx, fy, fz = fixation

    xs = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    ys = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    zs = (torch.arange(Z, dtype=torch.float32) + 0.5) / Z

    # L-inf distance from fixation for every voxel  (H, W, Z)
    dx = (xs - fx).abs()
    dy = (ys - fy).abs()
    dz = (zs - fz).abs()
    D = torch.max(dx[:, None, None],
                  torch.max(dy[None, :, None], dz[None, None, :]))

    foveal_mask = D <= foveal_r
    mid_mask    = (D > foveal_r) & (D <= mid_r)
    peri_mask   = D > mid_r

    def unique_blocks(mask, stride):
        idx = mask.nonzero()              # (N, 3) — one row per voxel in zone
        if idx.numel() == 0:
            return 0
        bx = idx[:, 0] // stride
        by = idx[:, 1] // stride
        bz = idx[:, 2] // stride
        nb_y = (W + stride - 1) // stride
        nb_z = (Z + stride - 1) // stride
        block_ids = bx * (nb_y * nb_z) + by * nb_z + bz
        return int(block_ids.unique().numel())

    foveal_tokens = int(foveal_mask.sum())   # stride=1: every voxel is its own token
    mid_tokens    = unique_blocks(mid_mask,  mid_stride)
    peri_tokens   = unique_blocks(peri_mask, peri_stride)

    return {
        'baseline':        H * W * Z,
        'foveal_voxels':   int(foveal_mask.sum()),
        'mid_voxels':      int(mid_mask.sum()),
        'peri_voxels':     int(peri_mask.sum()),
        'foveal_tokens':   foveal_tokens,
        'mid_tokens':      mid_tokens,
        'peri_tokens':     peri_tokens,
        'foveated_total':  foveal_tokens + mid_tokens + peri_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Analytical FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def _aligned_bound(center, radius, N, align):
    """Copy of FoveatedLocalAggregator._aligned_bound for crop-size computation."""
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
    lo = max(0, lo)
    return lo, hi


def analytical_flops(volume_shape, num_layer, num_channels, strides, c_in=128, fpn_out=128):
    """
    Closed-form MAC count for CustomResNet3D + 3-level GeneralizedLSSFPN (Conv3d).

    Convention: 1 FLOP = 1 MAC  (as commonly used in vision papers).
    Formula per Conv3d layer:  MACs = C_in × k³ × C_out × output_volume
    """
    H, W, Z = volume_shape
    V = H * W * Z

    # ── CustomResNet3D ────────────────────────────────────────────────────────
    backbone_macs = 0
    stage_volumes = []
    curr_c = c_in
    curr_v = V
    K3 = 27  # 3×3×3

    for n_blk, out_c, s in zip(num_layer, num_channels, strides):
        out_v = curr_v // (s ** 3)

        # Block 1: conv1 + conv2 + optional downsample (fires when s≠1 or channels differ)
        b1  = (curr_c * K3 * out_c) * out_v   # conv1
        b1 += (out_c  * K3 * out_c) * out_v   # conv2
        if s != 1 or curr_c != out_c:
            b1 += (curr_c * K3 * out_c) * out_v   # downsample conv

        # Remaining blocks (stride=1, no downsample)
        b_rest = 2 * (out_c * K3 * out_c) * out_v

        backbone_macs += b1 + (n_blk - 1) * b_rest
        stage_volumes.append(out_v)
        curr_c, curr_v = out_c, out_v

    # ── GeneralizedLSSFPN (top-down, num_outs=3) ──────────────────────────────
    # Level N-1 (smallest): lateral 1×1 (no concat) + FPN 3×3×3
    # Level i < N-1:        concat upsampled+backbone → lateral 1×1 + FPN 3×3×3
    fpn_macs = 0
    n_lvl = len(stage_volumes)

    for lvl in range(n_lvl - 1, -1, -1):   # top-down: highest→lowest
        vol  = stage_volumes[lvl]
        in_c = num_channels[lvl] if lvl == n_lvl - 1 else num_channels[lvl] + fpn_out

        fpn_macs += in_c  * 1  * fpn_out * vol   # 1×1×1 lateral conv
        fpn_macs += fpn_out * K3 * fpn_out * vol  # 3×3×3 FPN conv

    total_macs = backbone_macs + fpn_macs
    return {
        'backbone_gflops': backbone_macs / 1e9,
        'fpn_gflops':      fpn_macs      / 1e9,
        'total_gflops':    total_macs    / 1e9,
        'stage_volumes':   stage_volumes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Live Model FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def _live_flops(model, dummy):
    """Try fvcore, then thop. Returns (gflops, tool_name) or (None, None)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, dummy)
        if hasattr(fa, "unsupported_ops_settings"):
            fa.unsupported_ops_settings(warn=False, error=False)
        if hasattr(fa, "tracer_warnings"):
            fa.tracer_warnings("none")
        return fa.total() / 1e9, 'fvcore'
    except ImportError:
        pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs / 1e9, 'thop'
    except ImportError:
        pass
    return None, None


def _build_live_models(norm_cfg):
    """Instantiate FoveatedLocalAggregator and a dense-baseline wrapper."""
    from mmdet3d_plugin.models.tpvbranch.FoveatedLocalAggregator import FoveatedLocalAggregator

    aggregator = FoveatedLocalAggregator(
        local_encoder_backbone=dict(
            type='CustomResNet3D',
            numC_input=C,
            num_layer=NUM_LAYER,
            num_channels=NUM_CHANNELS,
            stride=STRIDE,
            norm_cfg=norm_cfg,
        ),
        local_encoder_neck=dict(
            type='GeneralizedLSSFPN',
            in_channels=NUM_CHANNELS,
            out_channels=C,
            start_level=0,
            num_outs=3,
            norm_cfg=norm_cfg,
            conv_cfg=dict(type='Conv3d'),
            act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='trilinear', align_corners=False),
        ),
        volume_h=VOLUME_H,
        volume_w=VOLUME_W,
        volume_z=VOLUME_Z,
        foveal_radius=FOVEAL_RADIUS,
        fixation=FIXATION,
    ).eval()

    class DenseBaseline(nn.Module):
        """Same backbone+neck run once on the full volume (no foveated split)."""
        def __init__(self, bb, nk):
            super().__init__()
            self.backbone = bb
            self.neck     = nk
        def forward(self, x):
            return self.neck(self.backbone(x))[0]

    dense = DenseBaseline(aggregator.backbone, aggregator.neck).eval()
    return aggregator, dense


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 68

    # ── 1. Token counts ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  1. TRANSFORMER TOKEN COUNTS")
    print(SEP)

    t = count_tokens(
        VOLUME_H, VOLUME_W, VOLUME_Z, FIXATION,
        FOVEAL_RADIUS, MID_RADIUS, MID_STRIDE, PERIPHERAL_STRIDE,
    )
    reduction = t['baseline'] / t['foveated_total']

    print(f"  Volume    : {VOLUME_H}×{VOLUME_W}×{VOLUME_Z} = {t['baseline']:,} voxels")
    print(f"  Fixation  : {FIXATION}")
    print(f"  Radii     : foveal ≤ {FOVEAL_RADIUS}  (stride 1),  "
          f"mid ≤ {MID_RADIUS}  (stride {MID_STRIDE}),  "
          f"peripheral  (stride {PERIPHERAL_STRIDE})")
    print()
    print(f"  {'Zone':<22} {'Voxels':>10}  {'Tokens':>10}  {'Compression':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline (no pooling)':<22} {t['baseline']:>10,}  {t['baseline']:>10,}  {'1.0×':>12}")
    print(f"  {'Foveal  (stride 1)':<22} {t['foveal_voxels']:>10,}  {t['foveal_tokens']:>10,}"
          f"  {'1.0×':>12}")
    print(f"  {'Mid     (stride '+str(MID_STRIDE)+')':<22} {t['mid_voxels']:>10,}  {t['mid_tokens']:>10,}"
          f"  {t['mid_voxels']/t['mid_tokens']:>11.1f}×")
    print(f"  {'Peripheral (stride '+str(PERIPHERAL_STRIDE)+')':<22} {t['peri_voxels']:>10,}  {t['peri_tokens']:>10,}"
          f"  {t['peri_voxels']/t['peri_tokens']:>11.1f}×")
    print(f"  {'-'*58}")
    print(f"  {'Foveated total':<22} {'':>10}   {t['foveated_total']:>10,}  {reduction:>11.1f}×")

    # ── 2. Analytical FLOPs ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  2. ANALYTICAL FLOPs  (1 FLOP = 1 MAC)")
    print(SEP)

    fx, fy, fz = FIXATION
    x0, x1 = _aligned_bound(fx, FOVEAL_RADIUS, VOLUME_H, ALIGN)
    y0, y1 = _aligned_bound(fy, FOVEAL_RADIUS, VOLUME_W, ALIGN)
    z0, z1 = _aligned_bound(fz, FOVEAL_RADIUS, VOLUME_Z, ALIGN)

    fov_shape  = (x1 - x0, y1 - y0, z1 - z0)
    glob_shape = (VOLUME_H // 2, VOLUME_W // 2, VOLUME_Z // 2)
    full_shape = (VOLUME_H, VOLUME_W, VOLUME_Z)

    base = analytical_flops(full_shape, NUM_LAYER, NUM_CHANNELS, STRIDE, C)
    fov  = analytical_flops(fov_shape,  NUM_LAYER, NUM_CHANNELS, STRIDE, C)
    glob = analytical_flops(glob_shape, NUM_LAYER, NUM_CHANNELS, STRIDE, C)

    fov_total   = fov['total_gflops'] + glob['total_gflops']
    flop_reduc  = base['total_gflops'] / fov_total

    print(f"  Foveal crop : x[{x0}:{x1}] y[{y0}:{y1}] z[{z0}:{z1}]"
          f"  →  {fov_shape}  (V={fov_shape[0]*fov_shape[1]*fov_shape[2]:,})")
    print(f"  Global input: 2× avg-pool  →  {glob_shape}  (V={glob_shape[0]*glob_shape[1]*glob_shape[2]:,})")
    print()
    print(f"  {'Stream':<34} {'Backbone':>9}  {'FPN':>7}  {'Total':>7}")
    print(f"  {'-'*62}")
    print(f"  {'Baseline (dense '+str(full_shape)+')':<34} "
          f"{base['backbone_gflops']:>8.1f}G  {base['fpn_gflops']:>6.1f}G  {base['total_gflops']:>6.1f}G")
    print(f"  {'Foveal stream '+str(fov_shape):<34} "
          f"{fov['backbone_gflops']:>8.1f}G  {fov['fpn_gflops']:>6.1f}G  {fov['total_gflops']:>6.1f}G")
    print(f"  {'Global stream '+str(glob_shape):<34} "
          f"{glob['backbone_gflops']:>8.1f}G  {glob['fpn_gflops']:>6.1f}G  {glob['total_gflops']:>6.1f}G")
    print(f"  {'-'*62}")
    print(f"  {'Foveated total (both streams)':<34} {'':>9}  {'':>7}  {fov_total:>6.1f}G")
    print(f"  {'Reduction':<34} {'':>9}  {'':>7}  {flop_reduc:>6.1f}×")

    # ── 3. Live model FLOPs ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. LIVE MODEL FLOPs  (fvcore / thop)")
    print(SEP)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        aggregator, dense = _build_live_models(norm_cfg)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        aggregator = aggregator.to(device)
        dense      = dense.to(device)
        dummy = torch.zeros(1, C, VOLUME_H, VOLUME_W, VOLUME_Z, device=device)

        gflops_fov,  tool = _live_flops(aggregator, dummy)
        gflops_dense, _   = _live_flops(dense, dummy)

        if tool is None:
            print("  No FLOPs library found.  Install one and rerun:")
            print("    pip install fvcore    # recommended")
            print("    pip install thop")
        else:
            print(f"  (using {tool})")
            print()
            if gflops_dense is not None:
                print(f"  Dense baseline (DenseBaseline):      {gflops_dense:>7.1f} GFLOPs")
            if gflops_fov is not None:
                print(f"  FoveatedLocalAggregator:             {gflops_fov:>7.1f} GFLOPs")
            if gflops_dense and gflops_fov:
                print(f"  Live reduction:                      {gflops_dense/gflops_fov:>7.1f}×")

    except ImportError as e:
        print(f"  Could not import project models: {e}")
        print("  Activate the project conda env and run from the VoxDet repo root.")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{SEP}\n")


if __name__ == '__main__':
    main()
