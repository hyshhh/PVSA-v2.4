from mmseg.registry import MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from .hys_Nex_21_6 import BiFormer
# from .biformer import BiFormer
from .bi_topp_vote import VTFormer
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import os
import warnings
import numpy as np
import cv2 
from PIL import Image
import torch
import torch.nn.functional as F


_TOPP_BRANCH_STAGE_LOGGED = set()


def _time_cuda_stage(enabled, tensor, fn):
    if not enabled or not torch.cuda.is_available() or not tensor.is_cuda:
        return fn(), None
    with torch.cuda.device(tensor.device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        return out, start.elapsed_time(end)


def _log_topp_branch_stage_debug(stage, x_shape, cnn_shape, out_shape, times):
    key = (stage, x_shape, cnn_shape, out_shape)
    if key in _TOPP_BRANCH_STAGE_LOGGED:
        return
    _TOPP_BRANCH_STAGE_LOGGED.add(key)
    parts = ' '.join(
        f'{name}={elapsed:.4f}ms' for name, elapsed in times.items())
    print(
        '[PVSA TopP Stage] '
        f'path=backbone_fusion stage={stage} '
        f'x={x_shape} cnn={cnn_shape} out={out_shape} {parts}')


@MODELS.register_module()
class BiFormer_fusion(VTFormer):
    def __init__(self,
                 pretrained=None,
                 use_topp_flash=False,
                 topp_flash_backend=None,
                 topp_flash_block_windows=64,
                 topp_flash_debug=False,
                 use_pruned_kv_gather=False,
                 feature_vis_config=None,
                 **kwargs):
        try:
            super().__init__(
                use_topp_flash=use_topp_flash,
                topp_flash_backend=topp_flash_backend,
                topp_flash_block_windows=topp_flash_block_windows,
                topp_flash_debug=topp_flash_debug,
                use_pruned_kv_gather=use_pruned_kv_gather,
                **kwargs)
        except TypeError as exc:
            flash_args = (
                'use_topp_flash',
                'topp_flash_backend',
                'topp_flash_block_windows',
                'topp_flash_debug',
                'use_pruned_kv_gather',
            )
            if not any(arg in str(exc) for arg in flash_args):
                raise
            if (use_topp_flash or topp_flash_backend is not None
                    or topp_flash_debug or use_pruned_kv_gather):
                warnings.warn(
                    '当前 VTFormer 不支持 Top-P Flash 参数，已降级为普通注意力路径。')
            super().__init__(**kwargs)
        self.topp_flash_debug = topp_flash_debug
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
        default_feature_vis_config = dict(
            enabled=False,
            save_dir='cam/features_imgs4',
            out_size=512,
            channel_reduce='mean')
        if feature_vis_config:
            default_feature_vis_config.update(feature_vis_config)
        self.feature_vis_config = default_feature_vis_config
        


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
        out = []
        cnn_encoder_out = x
        stage_profile = self.topp_flash_debug

        feature_vis_enabled = self.feature_vis_config.get('enabled', False)
        save_dir = self.feature_vis_config.get('save_dir', 'cam/features_imgs4')
        if feature_vis_enabled:
            os.makedirs(save_dir, exist_ok=True)
        channel1=[]
        channel2=[]
        channel3=[]

        def run_stage_timer(times, name, tensor, fn):
            result, elapsed = _time_cuda_stage(stage_profile, tensor, fn)
            if elapsed is not None:
                times[name] = elapsed
            return result

        for i in range(4):
            stage_times = {}
            stage_input_shape = tuple(x.shape)
            if feature_vis_enabled:
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_xinput.png')
            cnn_encoder_out = run_stage_timer(
                stage_times, 'cnn_branch', cnn_encoder_out,
                lambda i=i: self.downsample_layers2[i](cnn_encoder_out))
            x = run_stage_timer(
                stage_times, 'trans_down', x,
                lambda i=i: self.downsample_layers[i](x))
            x = run_stage_timer(
                stage_times, 'trans_stage', x,
                lambda i=i: self.stages[i](x))
            if feature_vis_enabled:
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_before_FAM_x.png')
                self._save_feature_channel_as_image(cnn_encoder_out, f'{save_dir}/stage{i}_before_FAM_cnn.png')

            x, cnn_encoder_out = run_stage_timer(
                stage_times, 'fam', x,
                lambda i=i: self.FAM[i](x, cnn_encoder_out))
            channel1.append(x)
            channel2.append(cnn_encoder_out)

            if feature_vis_enabled:
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_after_FAM_x.png')
                self._save_feature_channel_as_image(cnn_encoder_out, f'{save_dir}/stage{i}_after_FAM_cnn.png')
            if stage_times:
                _log_topp_branch_stage_debug(
                    i, stage_input_shape, tuple(cnn_encoder_out.shape),
                    tuple(x.shape), stage_times)

        for i in range(4):
            stage_times = {}
            fused = run_stage_timer(
                stage_times, 'fusion_conv', channel1[i],
                lambda i=i: self.fusion[i](
                    torch.cat((channel1[i], channel2[i]), dim=1)))
            channel3.append(fused)  # dim=1 表示按通道拼接
            if stage_times:
                _log_topp_branch_stage_debug(
                    f'fusion{i}', tuple(channel1[i].shape),
                    tuple(channel2[i].shape), tuple(fused.shape),
                    stage_times)
        for i in range(3):
            stage_times = {}
            C1 = run_stage_timer(
                stage_times, 'mask_conv1', channel1[i + 1],
                lambda i=i: self.conv11[i](channel1[i + 1]))
            C2 = run_stage_timer(
                stage_times, 'mask_conv2', channel1[i + 1],
                lambda i=i: self.conv12[i](channel1[i + 1]))
            bn_channel1 = run_stage_timer(
                stage_times, 'mask_bn1', C1,
                lambda i=i: self.sigmoid(self.bn[i](C1)))
            bn_channel2 = run_stage_timer(
                stage_times, 'mask_bn2', C2,
                lambda i=i: self.sigmoid(self.bn[i](C2)))
            if feature_vis_enabled and i==0:
                self._save_feature_channel_as_image(self.upsample2(bn_channel1), f'{save_dir}/mask1.png')
                self._save_feature_channel_as_image(self.upsample2(bn_channel2), f'{save_dir}/mask2.png')
            channel3[i] = run_stage_timer(
                stage_times, 'mask_fusion', channel3[i],
                lambda i=i: channel3[i] +
                self.upsample2(bn_channel1) * channel3[i] +
                self.upsample2(bn_channel2) * channel3[i])
            if stage_times:
                _log_topp_branch_stage_debug(
                    f'mask{i}', tuple(channel1[i + 1].shape),
                    tuple(channel3[i].shape), tuple(channel3[i].shape),
                    stage_times)

        for i in range(4):
            if feature_vis_enabled:
                self._save_feature_channel_as_image(channel3[i], f'{save_dir}/stage{i}_after_channel.png')
            stage_times = {}
            normed = run_stage_timer(
                stage_times, 'out_norm', channel3[i],
                lambda i=i: self.extra_norms[i](channel3[i]))
            out.append(normed)
            if stage_times:
                _log_topp_branch_stage_debug(
                    f'out{i}', tuple(channel3[i].shape),
                    tuple(channel3[i].shape), tuple(normed.shape),
                    stage_times)
        return tuple(out)



    def _save_feature_channel_as_image(
        self,
        feature_map,
        file_path,
        out_size=None,        # (H, W)，如 (512, 512)
        channel_reduce=None # "mean" | "max"
    ):
        """
        feature_map: [B, C, H, W] or [C, H, W]
        file_path: 保存路径
        out_size: 上采样到的空间尺寸 (H, W)，None 表示不变
        channel_reduce: 通道聚合方式
        """

        if out_size is None:
            out_size = self.feature_vis_config.get('out_size', 512)
        if channel_reduce is None:
            channel_reduce = self.feature_vis_config.get('channel_reduce', 'mean')

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
    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
