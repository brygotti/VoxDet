import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d_plugin.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, Focal_CE_ssc_loss, lovasz_softmax_loss, BLV_ssc_loss

@HEADS.register_module()
class OccHeadCLIP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=None,
        balance_cls_weight=True,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=None,
        train_cfg=None,
        test_cfg=None
    ):
        super(OccHeadCLIP, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx

        self.with_cp = with_cp
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        self.occ_size = occ_size
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] * 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i],
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=512, kernel_size=1, stride=1, padding=0),
            )
            self.occ_convs.append(occ_conv)
        txt_features = torch.from_numpy(np.load("text_features.npy")) 
        
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        # to buffer
        self.register_buffer('txt_feats', txt_features)
        
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
    
    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
                output_occs.append(torch.utils.checkpoint.checkpoint(occ_conv, feats))
            else:
                output_occs.append(occ_conv(feats))
        
        # to float 32self.txt_feats

        logits = torch.einsum("b c x y z, n c -> b n x y z", output_occs[0], self.txt_feats.to(torch.float32).to(output_occs[0].device))

        result = {
            'output_voxels': F.interpolate(logits, size=self.occ_size, mode='trilinear', align_corners=False).contiguous()
        }
        return result
    
    def loss(self, output_voxels, target_voxels):
        loss_dict = {}
        loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
    
@HEADS.register_module()
class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=None,
        balance_cls_weight=True,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=None,
        train_cfg=None,
        test_cfg=None,
        depth_weight_cfg=None, # on ajoute un nouveau parametre pour depth loss
    ):
        super(OccHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx

        self.with_cp = with_cp
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        self.occ_size = occ_size
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.train_cfg = train_cfg
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i],
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            )
            self.occ_convs.append(occ_conv)
        self.class_frequencies = class_frequencies
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

        # Nouvelle depth loss
        depth_weight_defaults = dict(
            enabled=False,
            mode='linear', # linear, exp, inverse
            depth_axis=0, # l’axe de profondeur dans le tenseur de sortie (0 pour [B, C, D, H, W], 1 pour [B, D, H, W])
            num_bins=8, # nombre de couche de profondeur
            min_weight=0.5, # poids minimum pour les couches les plus proches
            max_weight=1.5, # poids maximum pour les couches les plus lointaines
            normalize=True, # normaliser les poids pour garder une echelle de loss stable
        )

        # prends par default si pas de config fournie
        self.depth_weight_cfg = depth_weight_defaults if depth_weight_cfg is None else {**depth_weight_defaults, **depth_weight_cfg}

    # Decoupe un tenseur de sortie en tranches de profondeur selon l’axe indique, en gardant les dimensions de batch et de canaux intactes
    def _slice_volume(self, tensor, start, end, depth_axis):
        dim = depth_axis + 1 if tensor.dim() == 4 else depth_axis + 2 # on ajoute 1 pour ignorer le batch, (ou 1 de plus pour ignorer les canaux)
        index = [slice(None)] * tensor.dim() # on cree une liste d’index qui prend tout pour chaque dimension
        index[dim] = slice(start, end) # on remplace l’index de la dimension de profondeur par une tranche qui va de start a end
        return tensor[tuple(index)] # on convertit la liste d’index en tuple pour l’utiliser comme index de tenseur

    # Construit les poids de profondeur en fonction de la configuration. Elle retourne les bords des bins et les poids correspondants.
    def _build_depth_weights(self, depth_size, device):
        # check si depth weighting est enabled
        cfg = self.depth_weight_cfg
        if not cfg.get('enabled', False):
            return None

        # On cree des bins fixes, puis un poids scalaire pour chaque bin.
        num_bins = max(1, int(cfg['num_bins']))
        mode = cfg['mode']
        min_weight = float(cfg['min_weight'])
        max_weight = float(cfg['max_weight'])
        normalize = bool(cfg['normalize'])

        # On cree des bins de profondeur de taille egale, en arrondissant pour couvrir toute la profondeur. Le dernier bin peut etre plus grand si depth_size n'est pas divisible par num_bins.
        bin_edges = torch.linspace(0, depth_size, steps=num_bins + 1, device=device)
        bin_edges = torch.round(bin_edges).long()
        bin_edges[0] = 0
        bin_edges[-1] = depth_size

        # Centre normalise de chaque bin, entre 0 et 1.
        # rester entre 0 et 1 sert a garder une echelle de poids stable, et a permettre des fonctions de poids qui varient de maniere non lineaire avec la profondeur.
        centers = (torch.arange(num_bins, device=device, dtype=torch.float32) + 0.5) / num_bins

        # on aura qqch du style :
        # bin_edges = [0, 4, 8, 12, 16, 20, 24, 28, 32] pour depth_size=32 et num_bins=8
        # centers = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375] pour les centres normalises des bins

        # On calcule un poids pour chaque bin en fonction de son centre normalise, selon le mode choisi.
        if mode == 'linear':
            weights = min_weight + (max_weight - min_weight) * centers
        elif mode == 'exp':
            alpha = float(cfg.get('alpha', 2.0))
            weights = torch.exp(alpha * centers)
        elif mode == 'inverse':
            alpha = float(cfg.get('alpha', 4.0))
            weights = 1.0 / (1.0 + alpha * centers)
        else:
            raise ValueError(f'Unsupported depth weight mode: {mode}')

        if normalize:
            weights = weights / weights.mean().clamp_min(1e-6)

        return bin_edges, weights

    # Applique la depth loss
    def _depth_weighted_loss(self, loss_fn, output_tensor, target_tensor):
        cfg = self.depth_weight_cfg
        if not cfg.get('enabled', False):
            return loss_fn(output_tensor, target_tensor)

        
        depth_axis = int(cfg['depth_axis'])

        # On recalcul la depth size
        depth_size = output_tensor.shape[depth_axis + 2] if output_tensor.dim() == 5 else target_tensor.shape[depth_axis + 1]
        # On construit les bins de profondeur et les poids associes
        bin_edges, bin_weights = self._build_depth_weights(depth_size, output_tensor.device)

        # On va appliquer la loss sur chaque tranche de profondeur, puis faire une moyenne ponderee par les poids de profondeur.
        total_loss = output_tensor.new_tensor(0.0)
        total_weight = output_tensor.new_tensor(0.0)

        # On itere sur les bins de profondeur, en decoupant le tenseur de sortie et le tenseur cible en tranches correspondantes.
        for start, end, weight in zip(bin_edges[:-1], bin_edges[1:], bin_weights):
            start = int(start.item())
            end = int(end.item())
            if end <= start:
                continue

            output_slice = self._slice_volume(output_tensor, start, end, depth_axis)
            target_slice = self._slice_volume(target_tensor, start, end, depth_axis)
            # On calcule la meme fonction loss sur la tranche, puis on la pondere.
            total_loss = total_loss + weight * loss_fn(output_slice, target_slice)
            total_weight = total_weight + weight

        return total_loss / total_weight.clamp_min(1e-6)
    
    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

        
        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
   
                output_occs.append(torch.utils.checkpoint.checkpoint(occ_conv, feats))
            else:
                output_occs.append(occ_conv(feats))
        
        result = {
            'output_voxels': F.interpolate(output_occs[0], size=self.occ_size, mode='trilinear', align_corners=False).contiguous()
        }
        return result
    
    def loss(self, output_voxels, target_voxels):
        loss_dict = {}

        ce_loss = lambda out, tgt: CE_ssc_loss(out, tgt, self.class_weights.type_as(out), ignore_index=255)
        sem_loss = lambda out, tgt: sem_scal_loss(out, tgt, ignore_index=255)
        geo_loss = lambda out, tgt: geo_scal_loss(out, tgt, ignore_index=255, non_empty_idx=self.empty_idx)

        # On applique la depth loss

        # loss voxel ce = loss de classification standard
        loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * self._depth_weighted_loss(ce_loss, output_voxels, target_voxels)
        #loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        # loss voxel sem scal = loss de similarite semantique
        loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * self._depth_weighted_loss(sem_loss, output_voxels, target_voxels)
        #loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)

        # loss voxel geo scal = loss de similarite geometrique
        loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * self._depth_weighted_loss(geo_loss, output_voxels, target_voxels)
        #loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
    
        
        
