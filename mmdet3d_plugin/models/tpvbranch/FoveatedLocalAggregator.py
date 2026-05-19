"""
FoveatedLocalAggregator — two-stream 3D backbone with shared weights.

Design
------
Stream 1 (foveal): crops the foveal zone from the input voxel volume and
  processes it at FULL resolution through backbone + FPN neck.

Stream 2 (global): average-pools the full volume by 2 in each spatial dim,
  then processes the half-resolution volume through the SAME backbone + FPN.
  This covers the entire scene at reduced cost.

Merge: global output is upsampled back to full resolution via trilinear
  interpolation; the foveal region is then overwritten with the
  higher-resolution foveal features.

Result: a full (B, C, V_H, V_W, V_Z) feature map where the foveal zone
  has fine-grained features and the periphery has coarser features —
  mirroring the foveated tokenization already applied in VoxFormerHead.

FLOPs motivation (V_H=128, V_W=128, V_Z=16, C=128)
-------------------------------------------------------
  Dense backbone (original):   ~685 GFLOPs
  Foveated backbone (this):    ~151 GFLOPs  (4.5× reduction)

Key property: no extra parameters — backbone and neck are truly shared.
"""

import math
import torch
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS
from mmdet3d.models import builder
from mmcv.runner import BaseModule


@NECKS.register_module()
class FoveatedLocalAggregator(BaseModule):
    """Two-stream foveated 3D backbone (shared backbone + FPN neck weights).

    Args:
        local_encoder_backbone (dict): Config for CustomResNet3D.
        local_encoder_neck (dict): Config for GeneralizedLSSFPN.
        volume_h/w/z (int): Spatial dimensions of the input voxel grid.
        foveal_radius (float): L-inf radius of the foveal zone in
            normalised [0,1]^3 coordinates.
        fixation (tuple[float]): Fixation point (fx, fy, fz) in [0,1]^3.
            Must match the fixation used in VoxFormerHeadCrossAttention.
        align (int): Crop side lengths are padded to a multiple of this
            value so that stride-2 stages work cleanly.  Default: 4
            (supports up to 2 stride-2 stages).
    """

    def __init__(
        self,
        local_encoder_backbone,
        local_encoder_neck,
        volume_h: int,
        volume_w: int,
        volume_z: int,
        foveal_radius: float,
        fixation=(0.25, 0.5, 0.5),
        align: int = 4,
    ):
        super().__init__()
        self.backbone = builder.build_backbone(local_encoder_backbone)
        self.neck     = builder.build_neck(local_encoder_neck)

        # ── Pre-compute foveal crop bounds ─────────────────────────────────
        fx, fy, fz = fixation
        r = foveal_radius

        self.x0, self.x1 = _aligned_bound(fx, r, volume_h, align)
        self.y0, self.y1 = _aligned_bound(fy, r, volume_w, align)
        self.z0, self.z1 = _aligned_bound(fz, r, volume_z, align)

        crop_shape = (self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0)
        print(
            f"[FoveatedLocalAggregator] foveal crop: "
            f"x[{self.x0}:{self.x1}] y[{self.y0}:{self.y1}] z[{self.z0}:{self.z1}]"
            f" → {crop_shape}  ({100*crop_shape[0]*crop_shape[1]*crop_shape[2]/(volume_h*volume_w*volume_z):.1f}% of volume)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, V_H, V_W, V_Z) — full dense voxel feature volume.
        Returns:
            out: (B, C, V_H, V_W, V_Z) — foveated feature map.
        """
        _, _, V_H, V_W, V_Z = x.shape

        # ── Stream 1: foveal crop at full resolution ───────────────────────
        x_fov   = x[:, :, self.x0:self.x1, self.y0:self.y1, self.z0:self.z1]
        out_fov = self.neck(self.backbone(x_fov))[0]
        # out_fov: (B, C, x1-x0, y1-y0, z1-z0)

        # ── Stream 2: global context at half resolution ────────────────────
        x_glob   = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)
        out_glob = self.neck(self.backbone(x_glob))[0]
        # out_glob: (B, C, V_H/2, V_W/2, V_Z/2)

        # ── Merge (FoveaTer-style) ──────────────────────────────────────────
        # Tile each half-res cell to a 2×2×2 block via nearest-neighbour repeat.
        # Every voxel in a peripheral block gets the SAME pooled feature — no
        # smooth interpolation artefacts.  The downstream cross-attention head
        # (_pool_zone) then picks one leader per block; because all 8 voxels in
        # a block are identical, the average is trivially the block value itself.
        out_full = F.interpolate(
            out_glob, size=(V_H, V_W, V_Z),
            mode='nearest',
        )
        # 2. Paste exact foveal features (full-res, no pooling in this zone)
        out_full[:, :, self.x0:self.x1, self.y0:self.y1, self.z0:self.z1] = out_fov

        return out_full


# ─── helper ───────────────────────────────────────────────────────────────────

def _aligned_bound(center: float, radius: float, N: int, align: int):
    """Compute integer [lo, hi) crop bounds, padded to a multiple of align.

    The crop is centred around (center ± radius) in [0, 1] normalised coords,
    then extended symmetrically so that (hi - lo) is a multiple of align and
    the crop stays inside [0, N].
    """
    lo_f = (center - radius) * N
    hi_f = (center + radius) * N
    lo   = max(0, math.floor(lo_f))
    hi   = min(N, math.ceil(hi_f))

    size    = hi - lo
    aligned = math.ceil(size / align) * align
    extra   = aligned - size

    lo = max(0, lo - extra // 2)
    hi = lo + aligned
    if hi > N:
        hi = N
        lo = hi - aligned
    lo = max(0, lo)   # guard against negative after clamping
    return lo, hi
