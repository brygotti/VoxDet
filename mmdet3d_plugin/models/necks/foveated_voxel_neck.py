import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class FoveatedVoxelNeck(BaseModule):
    """3D foveated tokenization neck adapted from FoveaTer (ICLR 2023).

    Applies non-uniform average pooling to 3D voxel features based on
    distance from a fixation point (default: scene center). Voxels near
    the fixation receive high-resolution (no pooling) features; peripheral
    voxels receive progressively larger pooling kernels, reducing their
    effective spatial resolution analogously to the fovea/periphery split
    in biological vision.

    The input format matches SpatiallyDecoupledFPN output:
        (cls_feat_list, reg_feat_list), each a list of [B, C, X, Y, Z] tensors.
    The output preserves this format so VoxDetHead can consume it unchanged.

    Args:
        zone_radii (list[float]): Normalized L-inf distance thresholds
            defining zone boundaries, e.g. [0.3, 0.6] creates three zones.
            len(zone_radii) must equal len(pool_sizes) - 1.
        pool_sizes (list[int]): Pooling kernel size per zone (odd integers).
            pool_sizes[0] == 1 means the foveal zone is unmodified.
        fixation (list[float] | None): Normalized [fx, fy, fz] in [0,1]^3.
            Defaults to [0.5, 0.5, 0.5] (scene center).
        learnable_fixation (bool): If True, the fixation point is a learned
            parameter updated during training.
    """

    def __init__(
        self,
        zone_radii=(0.2, 0.4),
        pool_sizes=(1, 3, 5),
        strides=(1, 1, 1),
        upsample_mode='trilinear',
        fixation=None,
        learnable_fixation=False,
        soft_boundary=True,
        boundary_sharpness=20.0,
    ):
        super().__init__()
        assert len(pool_sizes) == len(zone_radii) + 1, (
            f"Expected len(pool_sizes)==len(zone_radii)+1, "
            f"got {len(pool_sizes)} vs {len(zone_radii)+1}"
        )
        assert len(strides) == len(pool_sizes), (
            f"Expected len(strides)==len(pool_sizes), "
            f"got {len(strides)} vs {len(pool_sizes)}"
        )
        assert strides[0] == 1, "Foveal zone (index 0) must always have stride 1."
        for k in pool_sizes:
            assert k % 2 == 1, f"pool_sizes must be odd integers, got {k}"
        self.soft_boundary = soft_boundary
        self.boundary_sharpness = boundary_sharpness
        self.strides = list(strides)
        self.upsample_mode = upsample_mode

        self.zone_radii = list(zone_radii)
        self.pool_sizes = list(pool_sizes)

        if fixation is None:
            fixation = [0.5, 0.5, 0.5]

        fix_tensor = torch.tensor(fixation, dtype=torch.float32)
        if learnable_fixation:
            self.fixation = nn.Parameter(fix_tensor)
        else:
            self.register_buffer("fixation", fix_tensor)

        # Cache the distance map keyed by spatial dims to avoid recomputing.
        self._dist_cache: dict = {}

    def _distance_map(self, X: int, Y: int, Z: int, device) -> torch.Tensor:
        """L-inf distance from fixation, normalized to [0, 1], shape [X, Y, Z]."""
        key = (X, Y, Z, str(device))
        if key in self._dist_cache and not self.fixation.requires_grad:
            return self._dist_cache[key]

        fx, fy, fz = self.fixation[0], self.fixation[1], self.fixation[2]

        gx = torch.arange(X, device=device).float() / max(X - 1, 1)  # [X]
        gy = torch.arange(Y, device=device).float() / max(Y - 1, 1)  # [Y]
        gz = torch.arange(Z, device=device).float() / max(Z - 1, 1)  # [Z]

        dx = (gx - fx).abs().view(X, 1, 1).expand(X, Y, Z)
        dy = (gy - fy).abs().view(1, Y, 1).expand(X, Y, Z)
        dz = (gz - fz).abs().view(1, 1, Z).expand(X, Y, Z)

        dist = torch.max(torch.max(dx, dy), dz)  # L-inf in [0, 1]

        if not self.fixation.requires_grad:
            self._dist_cache[key] = dist
        return dist

    def _foveate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply foveated non-uniform pooling to a single [B, C, X, Y, Z] tensor."""
        B, C, X, Y, Z = x.shape
        dist = self._distance_map(X, Y, Z, x.device)  # [X, Y, Z]

        min_dim = min(X, Y, Z)

        # Compute multi-scale pooled versions, all restored to [B, C, X, Y, Z].
        # stride=1: same-size neighbourhood averaging (original behaviour).
        # stride>1: genuine downsampling then trilinear/nearest upsample back,
        #           so peripheral voxels are computed from fewer unique values.
        scales = []
        for k, s in zip(self.pool_sizes, self.strides):
            k_eff = min(k, min_dim) if min_dim % 2 == 0 else min(k, min_dim - 1)
            k_eff = max(k_eff, 1)
            if k_eff % 2 == 0:
                k_eff -= 1  # keep odd

            if k_eff <= 1 and s == 1:
                scales.append(x)
            elif s == 1:
                scales.append(F.avg_pool3d(x, kernel_size=k_eff, stride=1,
                                           padding=k_eff // 2, count_include_pad=False))
            else:
                # Strided downsample — padding keeps coverage across the volume.
                x_down = F.avg_pool3d(x, kernel_size=k_eff, stride=s,
                                      padding=k_eff // 2, count_include_pad=False)
                align = False if self.upsample_mode == 'trilinear' else None
                x_up = F.interpolate(x_down, size=(X, Y, Z),
                                     mode=self.upsample_mode,
                                     align_corners=align)
                scales.append(x_up)

        # Build per-zone weight maps and blend scales.
        # soft_boundary=True: sigmoid transitions avoid hard edges at zone boundaries.
        # soft_boundary=False: binary masks (faster, may cause boundary artifacts).
        s = self.boundary_sharpness
        n = len(self.zone_radii)

        if self.soft_boundary:
            # gate[i](d) = sigmoid(s * (r_i - d)): probability of being in zone <= i
            # zone weight[i] = gate[i] - gate[i-1], with gate[-1]=0 and gate[n]=1
            gates = [torch.sigmoid(s * (dist - r)) for r in self.zone_radii]
            # gates[i] is high where dist > r_i  (peripheral side)
            weight_fovea = 1.0 - gates[0]                          # zone 0
            weight_periph = gates[-1]                               # last zone
            weights = [weight_fovea]
            for i in range(1, n):
                weights.append(gates[i - 1] - gates[i])
            weights.append(weight_periph)
        else:
            weights = []
            for i in range(len(self.pool_sizes)):
                if i == 0:
                    weights.append((dist <= self.zone_radii[0]).float())
                elif i == n:
                    weights.append((dist > self.zone_radii[-1]).float())
                else:
                    weights.append(
                        ((dist > self.zone_radii[i - 1]) & (dist <= self.zone_radii[i])).float()
                    )

        out = torch.zeros_like(x)
        for w, scale in zip(weights, scales):
            # w: [X, Y, Z] → broadcast to [B, C, X, Y, Z]
            out = out + w.unsqueeze(0).unsqueeze(0) * scale

        return out

    def forward(self, x):
        """
        Args:
            x: One of:
                - tuple (cls_feat_list, reg_feat_list) — SpatiallyDecoupledFPN format
                - list of tensors
                - single tensor [B, C, X, Y, Z]
        Returns:
            Same structure as input with foveated features applied.
        """
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], list):
            cls_feats, reg_feats = x
            return (
                [self._foveate(f) for f in cls_feats],
                [self._foveate(f) for f in reg_feats],
            )
        if isinstance(x, (list, tuple)):
            return [self._foveate(f) for f in x]
        return self._foveate(x)
