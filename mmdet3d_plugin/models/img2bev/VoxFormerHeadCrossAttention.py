# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import pdb
import torch
import numpy as np
import torch.nn as nn
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding


@HEADS.register_module()
class VoxFormerHeadCrossAttention(nn.Module):

    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        embed_dims,
        cross_transformer,
        foveal_radius=None,
        mid_radius=None,
        mid_stride=1,
        peripheral_stride=1,
        fixation=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims
        self.foveal_radius = foveal_radius
        self.mid_radius = mid_radius
        self.mid_stride = mid_stride
        self.peripheral_stride = peripheral_stride
        self.fixation = list(fixation)

        self.data_config = data_config
        self.cross_transformer = build_transformer(cross_transformer)

        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)
        self.mlp_lss = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
        self.mlp_prior = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims // 2), nn.LayerNorm(self.embed_dims // 2),
                                       nn.LeakyReLU(), nn.Linear(self.embed_dims // 2, self.embed_dims))

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(torch.arange(self.volume_h),
                                    torch.arange(self.volume_w),
                                    torch.arange(self.volume_z),
                                    indexing='ij')

        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat([(xv.reshape(-1, 1) + 0.5) / self.volume_h, (yv.reshape(-1, 1) + 0.5) / self.volume_w,
                            (zv.reshape(-1, 1) + 0.5) / self.volume_z],
                           dim=-1)

        return vox_coords, ref_3d

    def _pool_zone(self, zone_idx, stride, vq_mod, ref_3d_mod, vox_coords, device, pool_mode='avg'):
        """Spatial block pool: group tokens whose voxel coords fall in the
        same stride×stride×stride block, pool their features and average positions.
        pool_mode: 'avg' (mean) or 'max' (element-wise max) over features.
        Returns (leaders_idx, group_ids) where group_ids[i] maps zone_idx[i]
        to its group's position in the attention output."""
        n = len(zone_idx)
        if stride <= 1 or n == 0:
            return zone_idx, torch.arange(n, device=device)

        # Block index for each token based on spatial (x, y, z) voxel coordinates
        nb_y = (self.volume_w + stride - 1) // stride  # plain Python int division, fine
        nb_z = (self.volume_z + stride - 1) // stride
        bx = torch.div(vox_coords[zone_idx, 0], stride, rounding_mode='floor')
        by = torch.div(vox_coords[zone_idx, 1], stride, rounding_mode='floor')
        bz = torch.div(vox_coords[zone_idx, 2], stride, rounding_mode='floor')
        block_ids = bx * (nb_y * nb_z) + by * nb_z + bz  # [n] unique per spatial block

        # Sort tokens by block so we can group them contiguously
        sorted_order = torch.argsort(block_ids)
        sorted_zone  = zone_idx[sorted_order]
        sorted_bids  = block_ids[sorted_order]

        _, inverse_indices, counts = torch.unique(
            sorted_bids, return_inverse=True, return_counts=True
        )
        n_groups = len(counts)

        # Pool features and average positions into group buckets
        linear = vox_coords[sorted_zone, 3]
        feats  = vq_mod[linear].clone()      # [n, C]
        pos    = ref_3d_mod[linear].clone()  # [n, 3]

        inv_exp_c = inverse_indices.unsqueeze(1).expand(-1, feats.shape[1]).contiguous()
        if pool_mode == 'max':
            pooled_feats = torch.full((n_groups, feats.shape[1]), float('-inf'), device=device, dtype=feats.dtype)
            group_starts = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                counts.cumsum(0)[:-1],
            ])
            for group_idx in range(n_groups):
                start = group_starts[group_idx].item()
                end = start + counts[group_idx].item()
                pooled_feats[group_idx] = feats[start:end].max(dim=0).values
        else:
            pooled_feats = torch.zeros(n_groups, feats.shape[1], device=device, dtype=feats.dtype)
            pooled_feats.scatter_add_(0, inv_exp_c, feats)
            pooled_feats /= counts.float().unsqueeze(1)

        inv_exp_p = inverse_indices.unsqueeze(1).expand(-1, 3).contiguous()
        pooled_pos = torch.zeros(n_groups, 3, device=device, dtype=pos.dtype)
        pooled_pos.scatter_add_(0, inv_exp_p, pos)
        pooled_pos /= counts.float().unsqueeze(1)

        # First token of each block becomes the leader
        is_first = torch.cat([
            torch.ones(1, dtype=torch.bool, device=device),
            sorted_bids[1:] != sorted_bids[:-1],
        ])
        leaders = sorted_zone[is_first]

        # Write spatial-block averages to leader slots
        vq_mod[vox_coords[leaders, 3]]     = pooled_feats
        ref_3d_mod[vox_coords[leaders, 3]] = pooled_pos

        # group_ids[i] = group index for zone_idx[i], used for fill-back broadcast
        group_ids = torch.zeros(n, dtype=torch.long, device=device)
        group_ids[sorted_order] = inverse_indices

        return leaders, group_ids

    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)

    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None, **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device

        lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)
        lss_volume_flatten = self.mlp_lss(lss_volume_flatten)
        volume_queries = lss_volume_flatten

        # DEBUG: verify lss_volume and proposal match the configured voxel grid
        _n_vox = self.volume_h * self.volume_w * self.volume_z
        if lss_volume_flatten.shape[0] != _n_vox:
            raise RuntimeError(
                f"[VoxFormerHead] lss_volume spatial size {lss_volume_flatten.shape[0]} "
                f"!= volume_h*w*z={_n_vox} ({self.volume_h}x{self.volume_w}x{self.volume_z}). "
                f"lss_volume shape: {lss_volume.shape}"
            )
        if proposal.reshape(-1).shape[0] != _n_vox:
            raise RuntimeError(
                f"[VoxFormerHead] proposal.reshape(-1) size {proposal.reshape(-1).shape[0]} "
                f"!= volume_h*w*z={_n_vox}. proposal shape: {proposal.shape}"
            )

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)

        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()
        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)

        if self.foveal_radius is not None and len(unmasked_idx) > 0:
            # L-inf (Chebyshev) distance from fixation for each unmasked voxel
            fixation = torch.tensor(self.fixation, dtype=ref_3d.dtype, device=device)
            dists = (ref_3d[unmasked_idx] - fixation).abs().max(dim=-1).values

            foveal_unmasked = unmasked_idx[dists <= self.foveal_radius]
            beyond_foveal = ~(dists <= self.foveal_radius)

            if self.mid_radius is not None:
                mid_unmasked = unmasked_idx[beyond_foveal & (dists <= self.mid_radius)]
                peri_unmasked = unmasked_idx[beyond_foveal & (dists > self.mid_radius)]
            else:
                mid_unmasked = unmasked_idx[beyond_foveal]
                peri_unmasked = unmasked_idx.new_empty(0)

            # Pool mid (avg) and peripheral (avg) zones; write pooled queries AND positions to leader slots
            vq_mod     = volume_queries.clone()
            ref_3d_mod = ref_3d.clone()
            mid_leaders, mid_group_ids   = self._pool_zone(mid_unmasked,  self.mid_stride,        vq_mod, ref_3d_mod, vox_coords, device, pool_mode='avg')
            peri_leaders, peri_group_ids = self._pool_zone(peri_unmasked, self.peripheral_stride, vq_mod, ref_3d_mod, vox_coords, device, pool_mode='avg')

            active_idx     = torch.cat([foveal_unmasked, mid_leaders, peri_leaders])
            volume_queries = vq_mod
            ref_3d         = ref_3d_mod
            use_foveal = True
        else:
            foveal_unmasked = mid_unmasked = peri_unmasked = unmasked_idx.new_empty(0)
            mid_leaders = peri_leaders = unmasked_idx.new_empty(0)
            mid_group_ids = peri_group_ids = None
            active_idx = unmasked_idx
            use_foveal = False

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            volume_queries,
            self.volume_h,
            self.volume_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=active_idx,
            grid_length=None,
            bev_pos=None,
            #  bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs)

        vox_feats = torch.empty((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)

        if use_foveal:
            n_foveal = len(foveal_unmasked)
            n_mid_g  = len(mid_leaders)
            # Zone 1 — foveal: 1-to-1
            if n_foveal > 0:
                vox_feats_flatten[vox_coords[foveal_unmasked, 3]] = seed_feats[0][:n_foveal]
            # Zone 2 — mid: broadcast group output to all members
            if len(mid_unmasked) > 0:
                mid_out = seed_feats[0][n_foveal:n_foveal + n_mid_g]
                vox_feats_flatten[vox_coords[mid_unmasked, 3]] = mid_out[mid_group_ids]
            # Zone 3 — peripheral: broadcast group output to all members
            if len(peri_unmasked) > 0:
                peri_out = seed_feats[0][n_foveal + n_mid_g:]
                vox_feats_flatten[vox_coords[peri_unmasked, 3]] = peri_out[peri_group_ids]
        else:
            vox_feats_flatten[vox_coords[unmasked_idx, 3]] = seed_feats[0]

        # Empty voxels (proposal == 0) always use mlp_prior
        vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mlp_prior(lss_volume_flatten[masked_idx, :]).float()

        vox_feats = vox_feats_flatten.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        vox_feats = vox_feats.permute(3, 0, 1, 2).unsqueeze(0)

        return vox_feats
