from mmseg.registry import MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.fusion import fuse_conv_bn_eval
from .bi_topp_vote import VTFormer
from ..utils.topp_flash_kernel import _load_cuda_extension
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import os
import warnings
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
@MODELS.register_module()
class BiFormer_fusion(VTFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        self.extra_norms = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.conv12=nn.ModuleList()
        self.conv11=nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            self.bn.append(nn.BatchNorm2d(self.embed_dim[i]))
            self.conv12.append(nn.Conv2d(2*self.embed_dim[i],self.embed_dim[i],1,1,0))
            self.conv11.append(nn.Conv2d(2*self.embed_dim[i],self.embed_dim[i],1,1,0))
            
            
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        # self.pool2=nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool4=nn.AvgPool2d(kernel_size=4, stride=4)
        # self.pool8=nn.AvgPool2d(kernel_size=8, stride=8)
        # self.norm = nn.LayerNorm(normalized_shape=1)  # 根据实际维度调整
        self.sigmoid = nn.Sigmoid()
        


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print(f'Loading pretrained weights from {pretrained}')
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            raise TypeError(f'pretrained must be a str or None, but got {type(pretrained)}')


    def forward_features(self, x: torch.Tensor):
        if not self.training:
            self.optimize_for_inference()

        vis_cfg = self.feature_vis_config
        vis_enabled = vis_cfg.get('enabled', False)
        vis_once = vis_cfg.get('once', True)
        vis_dir = vis_cfg.get('save_dir', 'cam/features_imgs4')
        vis_out_size = vis_cfg.get('out_size', 512)
        vis_reduce = vis_cfg.get('channel_reduce', 'mean')

        if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
            os.makedirs(vis_dir, exist_ok=True)

        out = []
        cnn_encoder_out = x
        channel1=[]
        channel2=[]
        channel3=[]
        for i in range(4):
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_xinput.png', vis_out_size, vis_reduce)
            cnn_encoder_out = self.downsample_layers2[i](cnn_encoder_out)
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_before_FAM_x.png', vis_out_size, vis_reduce)
                self._save_feature_channel_as_image(cnn_encoder_out, f'{vis_dir}/stage{i}_before_FAM_cnn.png', vis_out_size, vis_reduce)

            x, cnn_encoder_out = self.FAM[i](x, cnn_encoder_out)
            channel1.append(x)
            channel2.append(cnn_encoder_out)

            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_after_FAM_x.png', vis_out_size, vis_reduce)
                self._save_feature_channel_as_image(cnn_encoder_out, f'{vis_dir}/stage{i}_after_FAM_cnn.png', vis_out_size, vis_reduce)

        for i in range(4):
            channel3.append(self.fusion[i](torch.cat((channel1[i], channel2[i]), dim=1)))
        for i in range(3):
            C1=self.conv11[i](channel1[i + 1])
            C2=self.conv12[i](channel2[i + 1])
            bn_channel1 = self.sigmoid(self.bn[i](C1))
            bn_channel2 = self.sigmoid(self.bn[i](C2))
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)) and i==0:
                self._save_feature_channel_as_image(self.upsample2(bn_channel1), f'{vis_dir}/mask1.png', vis_out_size, vis_reduce)
                self._save_feature_channel_as_image(self.upsample2(bn_channel2), f'{vis_dir}/mask2.png', vis_out_size, vis_reduce)
            channel3[i] = channel3[i] + self.upsample2(bn_channel1) * channel3[i] + self.upsample2(bn_channel2) * channel3[i]

        for i in range(4):
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(channel3[i], f'{vis_dir}/stage{i}_after_channel.png', vis_out_size, vis_reduce)
            out.append(self.extra_norms[i](channel3[i]))

        if vis_enabled:
            self._feature_vis_saved = True
        return tuple(out)



    def _save_feature_channel_as_image(
        self,
        feature_map,
        file_path,
        out_size=512,        # (H, W)，如 (512, 512)
        channel_reduce="mean" # "mean" | "max"
    ):
        """
        feature_map: [B, C, H, W] or [C, H, W]
        file_path: 保存路径
        out_size: 上采样到的空间尺寸 (H, W)，None 表示不变
        channel_reduce: 通道聚合方式
        """

        # ---------- 1. 维度统一 ----------
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # [C, H, W]

        assert feature_map.dim() == 3, "feature_map must be [C, H, W]"

        # ---------- 2. 通道聚合（深层特征必须做） ----------
        if channel_reduce == "mean":
            fmap = feature_map.mean(dim=0, keepdim=True)  # [1, H, W]
        elif channel_reduce == "max":
            fmap, _ = feature_map.max(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported channel_reduce: {channel_reduce}")

        fmap = fmap.unsqueeze(0)  # [1, 1, H, W]

        # ---------- 3. 上采样到目标分辨率 ----------
        if out_size is not None:
            fmap = F.interpolate(
                fmap,
                size=out_size,
                mode="bilinear",
                align_corners=False
            )
        # ---------- 4. 转 numpy ----------
        fmap = fmap[0, 0].detach().cpu().numpy()
        # ---------- 5. 归一化（用于可视化） ----------
        fmap = fmap - fmap.min()
        fmap = fmap / (fmap.max() + 1e-5)
        # ---------- 6. 轻度平滑（可选，但论文图更友好） ----------
        fmap = cv2.GaussianBlur(fmap, (3, 3), sigmaX=0.5, sigmaY=0.5)
        # ---------- 7. 映射为彩色热力图 ----------
        cmap = plt.get_cmap("viridis")
        img_color = (cmap(fmap)[:, :, :3] * 255).astype(np.uint8)
        # ---------- 8. 保存 ----------
        Image.fromarray(img_color).save(file_path)

    def optimize_for_inference(self):
        if self.training:
            return
        # Fallback guard: skip if parent already fused
        if getattr(self, '_inference_fused', False):
            return
        # Fuse parent (VTFormer) conv-bn layers
        super().optimize_for_inference()
        for idx in range(len(self.conv11)):
            bn = self.bn[idx]
            if isinstance(bn, nn.modules.batchnorm._BatchNorm) and not bn.training:
                self.conv11[idx] = fuse_conv_bn_eval(self.conv11[idx], bn)
                self.bn[idx] = nn.Identity()
        if getattr(self, 'topp_flash_backend', None) in ('cuda', 'cuda_forward'):
            _load_cuda_extension()

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
