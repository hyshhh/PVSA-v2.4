from mmseg.registry import MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.fusion import fuse_conv_bn_eval
# from .hys_Nex_21_6 import BiFormer
# from .biformer import BiFormer
from .bi_topp_vote import VTFormer
from ..utils.topp_flash_kernel import _load_cuda_extension
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import os
import warnings
import numpy as np
import cv2 
from PIL import Image
import torch.nn.functional as F


_TOPP_BRANCH_STAGE_LOGGED = set()


def _can_parallel_branches(x, cnn_x, stage_profile, feature_vis_enabled):
    return (not stage_profile and not feature_vis_enabled
            and torch.cuda.is_available() and x.is_cuda and cnn_x.is_cuda
            and x.device == cnn_x.device)


def _time_cuda_wall(enabled, tensor, fn):
    if not enabled or not torch.cuda.is_available() or not tensor.is_cuda:
        return fn(), None
    with torch.cuda.device(tensor.device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(tensor.device)
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
        f'{name}={int(elapsed)}' if name == 'blocks'
        else f'{name}={elapsed:.4f}ms'
        for name, elapsed in times.items())
    print(
        '[PVSA TopP Stage] '
        f'path=backbone_fusion stage={stage} '
        f'x={x_shape} cnn={cnn_shape} out={out_shape} {parts}')


def _run_with_optional_wall_time(enabled, tensor, times, name, fn):
    out, elapsed = _time_cuda_wall(enabled, tensor, fn)
    if elapsed is not None:
        times[name] = elapsed
    return out


@MODELS.register_module()
class BiFormer_fusion(VTFormer):
    def __init__(self,
                 pretrained=None,
                 topp_flash_backend=None,
                 topp_flash_block_windows=64,
                 topp_flash_debug=False,
                 use_pruned_kv_gather=False,
                 feature_vis_config=None,
                 mask_fusion_scale=0.5,
                 **kwargs):
        try:
            super().__init__(
                topp_flash_backend=topp_flash_backend,
                topp_flash_block_windows=topp_flash_block_windows,
                topp_flash_debug=topp_flash_debug,
                use_pruned_kv_gather=use_pruned_kv_gather,
                **kwargs)
        except TypeError as exc:
            flash_args = (
                'topp_flash_backend',
                'topp_flash_block_windows',
                'topp_flash_debug',
                'use_pruned_kv_gather',
            )
            if not any(arg in str(exc) for arg in flash_args):
                raise
            if (topp_flash_backend is not None or topp_flash_debug
                    or use_pruned_kv_gather):
                warnings.warn(
                    '当前 VTFormer 不支持 Top-P CUDA 参数，已降级为普通注意力路径。')
            super().__init__(**kwargs)
        self.topp_flash_debug = topp_flash_debug
        self.extra_norms = nn.ModuleList()
        self.bn11 = nn.ModuleList()
        self.bn12 = nn.ModuleList()
        self.conv12=nn.ModuleList()
        self.conv11=nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            self.bn11.append(nn.BatchNorm2d(self.embed_dim[i]))
            self.bn12.append(nn.BatchNorm2d(self.embed_dim[i]))
            self.conv12.append(nn.Conv2d(self.embed_dim[i],self.embed_dim[i],1,1,0))
            self.conv11.append(nn.Conv2d(self.embed_dim[i],self.embed_dim[i],1,1,0))
            
            
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.sigmoid = nn.Sigmoid()
        default_feature_vis_config = dict(
            enabled=False,
            save_dir='cam/features_imgs4',
            out_size=512,
            channel_reduce='mean')
        if feature_vis_config:
            default_feature_vis_config.update(feature_vis_config)
        self.feature_vis_config = default_feature_vis_config
        self.mask_fusion_scale = float(mask_fusion_scale)
        self.mask_residual_gates = nn.Parameter(torch.zeros(4))
        self._branch_inference_fused = False
        self._parallel_branch_streams = {}


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

    def optimize_for_inference(self):
        super().optimize_for_inference()
        if (self.training or self._branch_inference_fused
                or self._disable_inference_fusion):
            return
        for idx in range(len(self.conv11)):
            bn11 = self.bn11[idx]
            if isinstance(bn11, nn.modules.batchnorm._BatchNorm) and not bn11.training:
                self.conv11[idx] = fuse_conv_bn_eval(self.conv11[idx], bn11)
                self.bn11[idx] = nn.Identity()
            bn12 = self.bn12[idx]
            if isinstance(bn12, nn.modules.batchnorm._BatchNorm) and not bn12.training:
                self.conv12[idx] = fuse_conv_bn_eval(self.conv12[idx], bn12)
                self.bn12[idx] = nn.Identity()
        if self.topp_flash_backend in ('cuda', 'cuda_forward'):
            _load_cuda_extension()
        self._branch_inference_fused = True


    def forward_features(self, x: torch.Tensor):
        if not self.training:
            self.optimize_for_inference()
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

        def run_parallel_branches(stage_idx, trans_x, cnn_x):
            if not _can_parallel_branches(
                    trans_x, cnn_x, False, feature_vis_enabled):
                next_cnn = self.downsample_layers2[stage_idx](cnn_x)
                next_trans = self.downsample_layers[stage_idx](trans_x)
                next_trans = self.stages[stage_idx](next_trans)
                return next_trans, next_cnn

            trans_stream = torch.cuda.current_stream(trans_x.device)
            stream_key = (cnn_x.device.type, cnn_x.device.index, stage_idx)
            cnn_stream = self._parallel_branch_streams.get(stream_key)
            if cnn_stream is None:
                cnn_stream = torch.cuda.Stream(device=cnn_x.device)
                self._parallel_branch_streams[stream_key] = cnn_stream
            with torch.cuda.stream(cnn_stream):
                next_cnn = self.downsample_layers2[stage_idx](cnn_x)
            next_trans = self.downsample_layers[stage_idx](trans_x)
            next_trans = self.stages[stage_idx](next_trans)
            trans_stream.wait_stream(cnn_stream)
            next_cnn.record_stream(trans_stream)
            return next_trans, next_cnn

        for i in range(4):
            stage_times = {}
            stage_input_shape = tuple(x.shape)
            if feature_vis_enabled:
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_xinput.png')
            def run_stage_body(i=i):
                nonlocal x, cnn_encoder_out
                x, cnn_encoder_out = run_parallel_branches(
                    i, x, cnn_encoder_out)
                if i in self.fam_stages:
                    x, cnn_encoder_out = self.FAM[i](x, cnn_encoder_out)
                return x, cnn_encoder_out

            _, stage_wall = _time_cuda_wall(
                stage_profile, x, run_stage_body)
            if stage_wall is not None:
                stage_times['blocks'] = float(len(self.stages[i]))
                stage_times['stage_total_wall'] = stage_wall
            if feature_vis_enabled:
                self._save_feature_channel_as_image(x, f'{save_dir}/stage{i}_before_FAM_x.png')
                self._save_feature_channel_as_image(cnn_encoder_out, f'{save_dir}/stage{i}_before_FAM_cnn.png')
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
            fused = _run_with_optional_wall_time(
                stage_profile, channel1[i], stage_times, 'fusion_base',
                lambda i=i: channel1[i] + channel2[i])
            channel3.append(fused)  # dim=1 表示按通道拼接
            if stage_times:
                _log_topp_branch_stage_debug(
                    f'fusion{i}', tuple(channel1[i].shape),
                    tuple(channel2[i].shape), tuple(fused.shape),
                    stage_times)
        for i in range(4):
            stage_times = {}
            if i not in self.fusion_stages:
                continue
            if self.mask_source == 'branch_low':
                mask_source1 = channel1[i]
                mask_source2 = channel2[i]
            else:
                mask_source1 = channel3[i]
                mask_source2 = channel3[i]
            C1 = _run_with_optional_wall_time(
                stage_profile, mask_source1, stage_times, 'mask_conv1',
                lambda i=i: self.conv11[i](mask_source1))
            C2 = _run_with_optional_wall_time(
                stage_profile, mask_source2, stage_times, 'mask_conv2',
                lambda i=i: self.conv12[i](mask_source2))
            bn_channel1 = _run_with_optional_wall_time(
                stage_profile, C1, stage_times, 'mask_bn1',
                lambda i=i: self.sigmoid(self.bn11[i](C1)))
            bn_channel2 = _run_with_optional_wall_time(
                stage_profile, C2, stage_times, 'mask_bn2',
                lambda i=i: self.sigmoid(self.bn12[i](C2)))
            if feature_vis_enabled and i==0:
                self._save_feature_channel_as_image(bn_channel1, f'{save_dir}/mask1.png')
                self._save_feature_channel_as_image(bn_channel2, f'{save_dir}/mask2.png')
            channel3[i] = _run_with_optional_wall_time(
                stage_profile, channel3[i], stage_times, 'mask_fusion',
                lambda i=i: channel3[i] + (
                    self.mask_fusion_scale
                    * torch.tanh(self.mask_residual_gates[i])) * (
                    bn_channel1 * channel1[i] + bn_channel2 * channel2[i]))
            if stage_times:
                _log_topp_branch_stage_debug(
                    f'mask{i}', tuple(mask_source1.shape),
                    tuple(mask_source2.shape), tuple(channel3[i].shape),
                    stage_times)

        for i in range(4):
            if feature_vis_enabled:
                self._save_feature_channel_as_image(channel3[i], f'{save_dir}/stage{i}_after_channel.png')
            stage_times = {}
            normed = _run_with_optional_wall_time(
                stage_profile, channel3[i], stage_times, 'out_norm',
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
