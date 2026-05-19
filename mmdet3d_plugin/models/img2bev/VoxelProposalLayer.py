import os
import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from .modules.utils import Voxelization
import spconv.pytorch as spconv

@HEADS.register_module()
class VoxelProposalLayer(BaseModule):
    def __init__(
        self,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[256, 256, 32],
        data_config=None,
        init_cfg=None,
        **kwargs
    ):
        super(VoxelProposalLayer, self).__init__(init_cfg)

        self.data_config = data_config
        self.init_cfg = init_cfg
        self.voxelize = Voxelization(
            point_cloud_range=point_cloud_range, 
            spatial_shape=np.array(input_dimensions))
        
        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)

        self.input_dimensions = input_dimensions
        
    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    
    def depth2lidar(self, image_grid, depth, cam_params):
        b, _, h, w = depth.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params

        points = torch.cat([image_grid.repeat(b, 1, 1, 1), depth], dim=1) # [b, 3, h, w]
        points = points.view(b, 3, h * w).permute(0, 2, 1)

        # undo pos-transformation
        points = points - post_trans.view(b, 1, 3)
        points = torch.inverse(post_rots).view(b, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam to ego
        points = torch.cat([points[:, :, 0:2, :] * points[:, :, 2:3, :], points[:, :, 2:3, :]], dim=2)
        
        if intrins.shape[3] == 4:
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(b, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(b, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(b, 1, 3)

        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(b, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(b, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points
    
    def lidar2voxel(self, points, device):
        points_reshape = []
        batch_idx = []
        tensor = torch.ones((1,), dtype=torch.long).to(device)

        for i, pc in enumerate(points):
            points_reshape.append(pc)
            batch_idx.append(tensor.new_full((pc.shape[0],), i))
        
        points_reshape, batch_idx = torch.cat(points_reshape), torch.cat(batch_idx)

        unq, unq_inv = self.voxelize(points_reshape, batch_idx)

        return unq, unq_inv
    
    def forward(self, cam_params, img_metas):
        depth = img_metas['stereo_depth']
        points = self.depth2lidar(self.image_grid, depth, cam_params)
        unq, unq_inv = self.lidar2voxel(points, points.device)

        batch_size = int(unq[:, 0].max().item()) + 1

        if os.environ.get("VOXDET_DEBUG_BOUNDS"):
            _d = self.input_dimensions
            _bad = (
                (unq[:, 0] < 0) | (unq[:, 0] >= batch_size) |
                (unq[:, 1] < 0) | (unq[:, 1] >= _d[0]) |
                (unq[:, 2] < 0) | (unq[:, 2] >= _d[1]) |
                (unq[:, 3] < 0) | (unq[:, 3] >= _d[2])
            )
            if _bad.any():
                raise RuntimeError(
                    f"[VoxelProposalLayer] voxel indices out of bounds — "
                    f"batch=[{unq[:,0].min().item()}, {unq[:,0].max().item()}] vs [0,{batch_size}) "
                    f"x=[{unq[:,1].min().item()}, {unq[:,1].max().item()}] vs [0,{_d[0]}) "
                    f"y=[{unq[:,2].min().item()}, {unq[:,2].max().item()}] vs [0,{_d[1]}) "
                    f"z=[{unq[:,3].min().item()}, {unq[:,3].max().item()}] vs [0,{_d[2]})"
                )

        sparse_tensor = spconv.SparseConvTensor(
            torch.ones(unq.shape[0], dtype=torch.float32).view(-1, 1).to(points.device),
            unq.int(), spatial_shape=self.input_dimensions, batch_size=batch_size,
            )
        input = sparse_tensor.dense()

        return input