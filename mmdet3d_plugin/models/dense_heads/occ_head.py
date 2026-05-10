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
            assert txt_features.shape[0] == out_channel, \
                f"text_features has {txt_features.shape[0]} classes, but out_channel={out_channel}"
            # self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            self.class_weights = torch.ones(out_channel) / out_channel
    
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
        distance_weight_cfg=None, # on ajoute un nouveau parametre pour distance loss
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
            # self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            self.class_weights = torch.ones(out_channel) / out_channel

        # Nouvelle depth loss
        # ici depth cest pour la distance a la camera, qui correspond a l’axe X dans le format de sortie de VoxDet / SemanticKITTI.
        # L’idee est de donner plus de poids aux erreurs sur les voxels proches de la camera et moins de poids aux erreurs sur les voxels lointains.
        # On peut aussi experimenter avec differentes fonctions de poids (lineaire, exponentielle, inverse) pour voir ce qui marche le mieux.
        distance_weight_defaults = dict(
            enabled=False,
            mode='linear',      # linear, exp, inverse
            min_weight=0.5,
            max_weight=1.5,
            normalize=True,     # normaliser les poids pour garder une echelle de loss stable
            loss_type='risk',    # risk ou distance, pour experimenter avec differentes formulations de la loss ponderee

            # parametres risk-aware
            dynamic_classes=[1, 2, 3, 4, 5, 6, 7, 8],
            # SemanticKITTI occupancy classique:
            # 0 empty
            # 1 car
            # 2 bicycle
            # 3 motorcycle
            # 4 truck
            # 5 other-vehicle
            # 6 person
            # 7 bicyclist
            # 8 motorcyclist
            # Verifie quand meme ton mapping exact si ton dataset a ete remappe.

            dynamic_lambda=1.0,     # +100% sur les classes dynamiques si = 1.0
            boundary_lambda=1.0,    # +100% sur les frontieres si = 1.0
            uncertainty_lambda=0.5, # +50% max sur les voxels incertains si = 0.5

            use_distance_in_risk=True,
            use_dynamic_in_risk=True,
            use_boundary_in_risk=True,
            use_uncertainty_in_risk=True,
        )

        # prends par default si pas de config fournie
        self.distance_weight_cfg = (
            distance_weight_defaults
            if distance_weight_cfg is None
            else {**distance_weight_defaults, **distance_weight_cfg}
        )


    # Construit une weight map selon la distance forward.
    # target_voxels est suppose etre de forme [B, X, Y, Z].
    # On retourne un tenseur [1, X, 1, 1], broadcastable sur [B, X, Y, Z].
    def _build_forward_weight_map(self, target_voxels):
        cfg = self.distance_weight_cfg
        device = target_voxels.device

        # Dans VoxDet / SemanticKITTI, target_voxels = [B, X, Y, Z]
        # Donc X est l'axe de distance forward.
        x_size = target_voxels.shape[1] # combien il y a de positions sur l’axe forward

        mode = cfg.get('mode', 'linear')
        min_weight = float(cfg.get('min_weight', 0.5))
        max_weight = float(cfg.get('max_weight', 1.5))
        alpha = float(cfg.get('alpha', 2.0))
        normalize = bool(cfg.get('normalize', True))

        # Centre normalise de chaque voxel sur l'axe X, entre 0 et 1.
        # proche camera / ego vehicule -> proche de 0
        # loin devant -> proche de 1
        centers = (torch.arange(x_size, device=device, dtype=torch.float32) + 0.5) / x_size

        # On calcule un poids pour chaque position X en fonction de sa distance forward.
        # Les voxels proches de la camera auront des poids plus eleves, les voxels lointains auront des poids plus faibles. sauf inverse
        
        if mode == 'linear':
            weights = max_weight + (min_weight - max_weight) * centers

        elif mode == 'exp':
            alpha = float(cfg.get('alpha', 2.0))
            weights = torch.exp(-alpha * centers)

        elif mode == 'inverse':
            alpha = float(cfg.get('alpha', 4.0))
            weights = 1.0 / (1.0 + alpha * centers)

        else:
            raise ValueError(f'Unsupported distance weight mode: {mode}')

        if normalize:
            weights = weights / weights.mean().clamp_min(1e-6)

        # Shape [1, X, 1, 1] pour pouvoir multiplier une loss [B, X, Y, Z]
        return weights.view(1, x_size, 1, 1)

    # Construit un masque pour les classes dynamiques.
    # target_voxels est suppose etre [B, X, Y, Z].
    # On retourne un masque float [B, X, Y, Z].
    def _build_dynamic_mask(self, target_voxels):
        cfg = self.distance_weight_cfg
        dynamic_classes = cfg.get('dynamic_classes', [1, 2, 3, 4, 5, 6, 7, 8])

        dynamic_mask = torch.zeros_like(target_voxels, dtype=torch.bool)

        for cls_id in dynamic_classes:
            dynamic_mask |= (target_voxels == int(cls_id))

        return dynamic_mask.float()

    # Construit un masque de frontiere dans la grille voxel.
    # Un voxel est considere comme frontiere si au moins un voisin 6-connecte a une classe differente.
    # On ignore les comparaisons avec le label 255.
    def _build_boundary_mask(self, target_voxels):
        valid = target_voxels != 255
        boundary = torch.zeros_like(target_voxels, dtype=torch.bool)

        # Differences selon X
        diff_x = (
            (target_voxels[:, 1:, :, :] != target_voxels[:, :-1, :, :])
            & valid[:, 1:, :, :]
            & valid[:, :-1, :, :]
        )
        boundary[:, 1:, :, :] |= diff_x
        boundary[:, :-1, :, :] |= diff_x

        # Differences selon Y
        diff_y = (
            (target_voxels[:, :, 1:, :] != target_voxels[:, :, :-1, :])
            & valid[:, :, 1:, :]
            & valid[:, :, :-1, :]
        )
        boundary[:, :, 1:, :] |= diff_y
        boundary[:, :, :-1, :] |= diff_y

        # Differences selon Z
        diff_z = (
            (target_voxels[:, :, :, 1:] != target_voxels[:, :, :, :-1])
            & valid[:, :, :, 1:]
            & valid[:, :, :, :-1]
        )
        boundary[:, :, :, 1:] |= diff_z
        boundary[:, :, :, :-1] |= diff_z

        return boundary.float()

    # Construit une carte d'incertitude du modele.
    # Plus l'entropie est grande, plus le modele hesite.
    # On detach pour eviter que le modele apprenne a manipuler directement cette pondération.
    def _build_uncertainty_weight(self, output_voxels):
        cfg = self.distance_weight_cfg
        uncertainty_lambda = float(cfg.get('uncertainty_lambda', 0.5))

        probs = torch.softmax(output_voxels.detach(), dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)

        # Normalisation par log(C), pour avoir une entropie environ entre 0 et 1.
        num_classes = output_voxels.shape[1]
        entropy = entropy / np.log(num_classes)

        uncertainty_weight = 1.0 + uncertainty_lambda * entropy

        return uncertainty_weight
    
    # Applique une CE ponderee par distance forward.
    # On garde la CE qui etait la de base, mais on multiplie chaque voxel par un poids qui depend de sa distance forward.
    # output_voxels est suppose etre [B, C, X, Y, Z].
    # target_voxels est suppose etre [B, X, Y, Z].
    def _forward_distance_ce_loss(self, output_voxels, target_voxels):
        cfg = self.distance_weight_cfg

        # Si la pondération par distance n'est pas activee, on garde la loss originale.
        if not cfg.get('enabled', False):
            return CE_ssc_loss(
                output_voxels,
                target_voxels,
                self.class_weights.type_as(output_voxels),
                ignore_index=255
            )

        # Cross entropy voxel-wise, sans reduction.
        # La sortie est [B, X, Y, Z], donc une loss par voxel.
        voxel_ce = F.cross_entropy(
            output_voxels,
            target_voxels.long(),
            weight=self.class_weights.type_as(output_voxels),
            ignore_index=255,
            reduction='none' # permet de garder une loss par voxel pour appliquer la pondération ensuite
        )

        # On construit les poids selon l'axe X.
        distance_weights = self._build_forward_weight_map(target_voxels)

        # On ignore les voxels avec label 255. (comme dans loriginal)
        valid_mask = (target_voxels != 255).float()

        # On applique la pondération par distance.
        weighted_loss = voxel_ce * distance_weights * valid_mask

        # Moyenne ponderee uniquement sur les voxels valides.
        normalizer = (distance_weights * valid_mask).sum().clamp_min(1e-6)

        return weighted_loss.sum() / normalizer

    # Nouvelle loss plus complexe:
    # Risk-aware foveated CE = distance + classes dynamiques + frontieres + incertitude.
    # L'idee est de donner plus de poids aux zones critiques:
    #   - proches devant
    #   - objets dynamiques
    #   - frontieres entre classes / objet-fond
    #   - voxels ou le modele est incertain
    def _risk_aware_foveated_ce_loss(self, output_voxels, target_voxels):
        cfg = self.distance_weight_cfg

        voxel_ce = F.cross_entropy(
            output_voxels,
            target_voxels.long(),
            weight=self.class_weights.type_as(output_voxels),
            ignore_index=255,
            reduction='none'
        )

        valid_mask = (target_voxels != 255).float()

        # Poids initialise a 1 partout.
        final_weight = torch.ones_like(target_voxels, dtype=torch.float32, device=target_voxels.device)

        # 1) Poids distance forward.
        if cfg.get('use_distance_in_risk', True):
            distance_weight = self._build_forward_weight_map(target_voxels)
            final_weight = final_weight * distance_weight

        # 2) Poids objets dynamiques.
        # Exemple: car, truck, bicycle, person, etc.
        if cfg.get('use_dynamic_in_risk', True):
            dynamic_lambda = float(cfg.get('dynamic_lambda', 1.0))
            dynamic_mask = self._build_dynamic_mask(target_voxels)
            dynamic_weight = 1.0 + dynamic_lambda * dynamic_mask
            final_weight = final_weight * dynamic_weight

        # 3) Poids frontieres.
        # Les frontieres sont importantes car elles definissent mieux la geometrie des objets.
        if cfg.get('use_boundary_in_risk', True):
            boundary_lambda = float(cfg.get('boundary_lambda', 1.0))
            boundary_mask = self._build_boundary_mask(target_voxels)
            boundary_weight = 1.0 + boundary_lambda * boundary_mask
            final_weight = final_weight * boundary_weight

        # 4) Poids incertitude.
        # Les voxels ou le modele hesite recoivent plus de poids.
        if cfg.get('use_uncertainty_in_risk', True):
            uncertainty_weight = self._build_uncertainty_weight(output_voxels)
            final_weight = final_weight * uncertainty_weight

        # On ignore les voxels invalides.
        final_weight = final_weight * valid_mask

        # Normalisation optionnelle pour garder une echelle de loss stable.
        if cfg.get('normalize', True):
            valid_mean = final_weight[valid_mask.bool()].mean().clamp_min(1e-6)
            final_weight = final_weight / valid_mean

        weighted_loss = voxel_ce * final_weight * valid_mask

        normalizer = (final_weight * valid_mask).sum().clamp_min(1e-6)

        return weighted_loss.sum() / normalizer


    # Fonction principale qui choisit quelle CE utiliser.
    # loss_type:
    #   none     -> CE originale
    #   distance -> CE ponderee par distance forward
    #   risk     -> CE risk-aware foveated
    def _selected_ce_loss(self, output_voxels, target_voxels):
        cfg = self.distance_weight_cfg

        if not cfg.get('enabled', False):
            return CE_ssc_loss(
                output_voxels,
                target_voxels,
                self.class_weights.type_as(output_voxels),
                ignore_index=255
            )

        loss_type = cfg.get('loss_type', 'distance')

        if loss_type == 'none':
            return CE_ssc_loss(
                output_voxels,
                target_voxels,
                self.class_weights.type_as(output_voxels),
                ignore_index=255
            )

        elif loss_type == 'distance':
            return self._forward_distance_ce_loss(output_voxels, target_voxels)

        elif loss_type == 'risk':
            return self._risk_aware_foveated_ce_loss(output_voxels, target_voxels)

        else:
            raise ValueError(f'Unsupported CE loss_type: {loss_type}')

    
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

        # On applique la CE choisie:
        #   - originale
        #   - distance forward
        #   - risk-aware foveated
        loss_dict['loss_voxel_ce'] = (
            self.loss_voxel_ce_weight
            * self._selected_ce_loss(output_voxels, target_voxels)
        )

        # Pour les 2 autres losses, on applique pas la ponderation par distance / risk.
        # Elles calculent des statistiques globales sur les classes / geometrie.
        # Les ponderer voxel-wise peut changer leur sens.
        loss_dict['loss_voxel_sem_scal'] = (
            self.loss_voxel_sem_scal_weight
            * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        )

        loss_dict['loss_voxel_geo_scal'] = (
            self.loss_voxel_geo_scal_weight
            * geo_scal_loss(
                output_voxels,
                target_voxels,
                ignore_index=255,
                non_empty_idx=self.empty_idx
            )
        )

        return loss_dict
    
        
        
