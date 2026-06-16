import os
from typing import Dict, Optional, Tuple
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .topp_flash_kernel import (can_run_topp_route_cuda,
                                is_topp_flash_available,
                                topp_flash_attention, topp_route_cuda,
                                warn_topp_route_cuda_fallback)


DEFAULT_ATTN_VIS_CONFIG = dict(enabled=False)
_TOPP_FLASH_STAGE_LOGGED = set()


def _normalize_route_configs(route_configs: Optional[Dict]) -> Dict[int, Dict]:
    if not route_configs:
        raise ValueError(
            'topp_route_configs must be provided by config when using '
            'ToppAttention.')
    return {int(key): dict(value) for key, value in route_configs.items()}


def _normalize_attn_vis_config(attn_vis_config: Optional[Dict]) -> Dict:
    config = DEFAULT_ATTN_VIS_CONFIG.copy()
    if attn_vis_config:
        config.update(attn_vis_config)
    return config


def _time_cuda_stage(enabled: bool, tensor: Tensor, fn):
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


def _log_topp_stage_debug(path: str, x: Tensor, q_pix: Tensor, kv_pix: Tensor,
                          r_idx: Tensor, times: Dict[str, float],
                          num_heads: int, qk_dim: int, dim: int,
                          n_win: int) -> None:
    route_shape = tuple(r_idx.shape) if hasattr(r_idx, 'shape') else tuple(r_idx)
    key = (
        path, tuple(x.shape), tuple(q_pix.shape), tuple(kv_pix.shape),
        route_shape, num_heads, qk_dim, dim, n_win)
    if key in _TOPP_FLASH_STAGE_LOGGED:
        return
    _TOPP_FLASH_STAGE_LOGGED.add(key)
    parts = (
        ' '.join(f'{name}={elapsed:.4f}ms'
                 for name, elapsed in times.items())
        if times else 'timing=off')
    print(
        '[PVSA TopP Stage] '
        f'path={path} x={tuple(x.shape)} q={tuple(q_pix.shape)} '
        f'kv={tuple(kv_pix.shape)} route={route_shape} '
        f'heads={num_heads} qk_dim={qk_dim} dim={dim} n_win={n_win} '
        f'{parts}')

def overlay_topk_attn_block_no_heatmap(img, topk_score, topk_index, total_patches, dark_ratio=0.3):
    """
    无热力图版本的 Top-K 注意力可视化 (聚光灯效果)

    Args:
        img: 图像路径或 numpy 数组
        topk_score: 1D tensor, 某个 query 对应的 top-k 得分
        topk_index: 1D tensor, 某个 query 对应的 top-k 索引
        total_patches: 原始序列的总长度
        dark_ratio: 背景变暗的比例 (0.0全黑, 1.0完全不变，0.3表示背景只有30%亮度)
    """
    if isinstance(img, str):
        img = cv2.imread(img)

    H, W = img.shape[:2]

    # 1. 获取索引
    index = topk_index.detach().cpu().numpy()

    # 2. 重建掩码网格 (不需要真实得分了，选中的地方设为 1，其余为 0)
    attn_reconstructed = np.zeros(total_patches, dtype=np.uint8)
    attn_reconstructed[index] = 1

    # 3. Reshape 为 2D 网格
    size = int(total_patches ** 0.5)
    attn_map = attn_reconstructed.reshape(size, size)

    # 4. 放大到图像尺寸 (Block 形式)
    scale_h = H // size
    scale_w = W // size
    attn_map_block = np.repeat(np.repeat(attn_map, scale_h, axis=0), scale_w, axis=1)

    # 5. Padding 补齐边界
    pad_h = H - attn_map_block.shape[0]
    pad_w = W - attn_map_block.shape[1]
    attn_map_block = np.pad(
        attn_map_block,
        ((0, pad_h), (0, pad_w)),
        mode='edge'
    )

    # 6. 生成聚光灯效果
    # 将 mask 扩展为 3 通道，以便与图像计算
    mask = attn_map_block[..., np.newaxis]

    # 生成一张变暗的底图
    dark_img = (img * dark_ratio).astype(np.uint8)

    # 核心：如果是 Top-K 区域 (mask==1)，使用原图；否则使用变暗的底图
    overlay = np.where(mask == 1, img, dark_img)

    return overlay
def overlay_attn_block(img, attn):
    if isinstance(img, str):
        img = cv2.imread(img)

    H, W = img.shape[:2]

    attn = attn.detach().cpu().numpy()
    size = int(len(attn) ** 0.5)

    attn_map = attn.reshape(size, size)

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

    scale_h = H // size
    scale_w = W // size

    # repeat 放大
    attn_map_block = np.repeat(np.repeat(attn_map, scale_h, axis=0), scale_w, axis=1)

    # ✅ 关键：padding补齐
    pad_h = H - attn_map_block.shape[0]
    pad_w = W - attn_map_block.shape[1]
    attn_map_block = np.pad(
        attn_map_block,
        ((0, pad_h), (0, pad_w)),
        mode='edge'   # 用边界值填充（最合理）
    )

    heatmap = cv2.applyColorMap((attn_map_block * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)

    return overlay
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False,
                 diff_routing=False, W=False, route_configs=None,
                 attn_vis_config=None, debug_route=False):
        super().__init__()
        self.route_flag = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.W=W
        self.debug_route = debug_route
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activate
        self.routing_act = nn.Softmax(dim=-1)
        self.flag=0
        route_configs = _normalize_route_configs(route_configs)
        if self.route_flag not in route_configs:
            raise KeyError(
                f'topk flag {self.route_flag} is not configured in '
                'topp_route_configs.')
        route_config = route_configs[self.route_flag]
        self.topk = int(route_config['maxk'])
        self.P = float(route_config['p'])
        self.Temperature = float(route_config['temperature'])
        self.energy = float(route_config['energy'])
        self.attn_vis_config = _normalize_attn_vis_config(attn_vis_config)
        self._attn_vis_saved = False

        self.silu=True
        #能量补偿因子

#top—p-v3_2025_12_25
    def forward(self, query: Tensor, key: Tensor,GA):

        if self.W==False or GA==None:
            if not self.diff_routing:
                query = query.detach()
                key = key.detach()
            # 1️⃣ Linear embedding
            # q = self.emb(query)    # (n, p2, c)
            # k = self.emb(key)      # (n, p2, c)
            q = F.normalize(query, dim=-1)
            k = F.normalize(query, dim=-1)
            attn = (q * self.scale) @ k.transpose(-2, -1)   # (n, p2, p2)
            # print(1)
        else:
            attn = GA
            print("HH",attn[0][1])
            print(2)
        # 3️⃣ Top-k selection (no sorting for speed)
        topk_score, topk_index = torch.topk(
            attn, k=self.topk, dim=-1, sorted=True
        )  # (n, p2, k)

        topk_score = torch.softmax(topk_score /self.Temperature, dim=-1)
        self._maybe_save_attention(attn, topk_score, topk_index)

        # 5️⃣ Cumulative probability pruning
        cumsum = torch.cumsum(topk_score, dim=-1)      # (n, p2, k)

        keep_mask = cumsum <= self.P                   # (n, p2, k)
        keep_len = keep_mask.sum(dim=-1, keepdim=True) # (n, p2, 1)
        keep_len = keep_len.clamp(min=1)

        # 6️⃣ Vectorized truncation (NO LOOP)
        max_len = keep_len.max()

        if self.debug_route:
            print(f'[Route] flag={self.route_flag} maxk={self.topk} p={self.P} '
                  f'temp={self.Temperature} energy={self.energy} '
                  f'max_len={max_len.item()} keep_len_min={keep_len.min().item():.0f} '
                  f'keep_len_mean={keep_len.float().mean().item():.1f}')

        # truncate to max_len
        topk_score = topk_score[..., :max_len]
        topk_index = topk_index[..., :max_len]
        if self.flag==3 and self.topk==4:
            print("第2",topk_score[0][0])
            print("第2'",topk_index[0][0])

        # position mask
        pos = torch.arange(max_len, device=topk_score.device)
        valid_mask = pos[None, None, :] < keep_len

        # apply mask
        topk_score = topk_score * valid_mask.to(topk_score.dtype)
        topk_index = topk_index.masked_fill(~valid_mask, 0)

        #能量补偿
        topk_score = topk_score * self.energy

        if self.flag==4 and self.topk==4:
            print("第3",topk_score[0][0])
            print("第3",topk_index[0][0])

        # 7️⃣ Final routing activation
        # r_weight = self.routing_act(topk_score)
        # if self.flag==0:
        #     print("第4",r_weight[0][0])

        return topk_score, topk_index, valid_mask

    def _maybe_save_attention(self, attn: Tensor, topk_score: Tensor,
                              topk_index: Tensor) -> None:
        config = self.attn_vis_config
        if not config.get('enabled', False):
            return
        if config.get('once', True) and self._attn_vis_saved:
            return
        trigger_maxk = config.get('trigger_maxk')
        if trigger_maxk is not None and self.topk != int(trigger_maxk):
            return

        if 'query_index' not in config:
            warnings.warn('attention visualize query_index is not configured.')
            return
        query_index = int(config['query_index'])
        if query_index >= attn.size(1):
            warnings.warn(
                f'attention visualize query_index={query_index} exceeds '
                f'available windows={attn.size(1)}.')
            return

        img_path = config.get('image_path')
        if not img_path:
            warnings.warn('attention visualize image_path is empty.')
            return

        if config.get('save_heatmap', False):
            overlay = overlay_attn_block(img_path, attn[0][query_index])
            save_path = config.get('heatmap_save_path')
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_path, overlay)

        if config.get('save_topk', False):
            overlay_topk = overlay_topk_attn_block_no_heatmap(
                img=img_path,
                topk_score=topk_score[0][query_index],
                topk_index=topk_index[0][query_index],
                total_patches=attn.shape[-1],
                dark_ratio=float(config.get('dark_ratio', 1.0)))
            save_path = config.get('topk_save_path')
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_path, overlay_topk)

        self._attn_vis_saved = True

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )
        # print("KV形状",topk_kv[0][0][0][0][0])
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
            # print("KV形状",topk_kv[0][0][0][0][0])
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v


class ToppAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim,num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=True,
                 side_dwconv=3,
                 auto_pad=False,W=False, use_topp_flash=False,
                 topp_flash_block_windows=64, topp_flash_backend=None,
                 use_pruned_kv_gather=False, pruned_kv_num_groups=1,
                 topp_route_configs=None,
                 attn_vis_config=None,
                 use_fast_attention=False,
                 debug_route=False,
                 topp_flash_debug=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.W=W
        self.use_topp_flash = use_topp_flash
        self.topp_flash_block_windows = topp_flash_block_windows
        self.topp_flash_backend = topp_flash_backend
        self.use_pruned_kv_gather = use_pruned_kv_gather
        self.pruned_kv_num_groups = pruned_kv_num_groups
        self.topp_route_configs = topp_route_configs
        self.attn_vis_config = attn_vis_config
        self.use_fast_attention = use_fast_attention
        self.topp_flash_debug = topp_flash_debug
        self._topp_flash_warned = False

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = True
        self.W=W
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,

                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing,W=self.W,
                                  route_configs=self.topp_route_configs,
                                  attn_vis_config=self.attn_vis_config,
                                  debug_route=debug_route)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x,GA, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        # NOTE: use padding for semantic segmentation
        # 输入填充处理
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################
        stage_debug = self.topp_flash_debug and not ret_attn_mask
        stage_profile = stage_debug
        stage_times = {}

        def run_stage(name, fn):
            out, elapsed = _time_cuda_stage(stage_profile, x, fn)
            if elapsed is not None:
                stage_times[name] = elapsed
            return out

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = run_stage('qkv', lambda: self.qkv(x))

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = run_stage(
            'kv_down',
            lambda: self.kv_down(
                rearrange(kv, 'n p2 h w c -> (n p2) c h w')))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # 对值部分应用深度可分离卷积作为位置编码
        lepe = run_stage(
            'lepe',
            lambda: self.lepe(
                rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                          i=self.n_win).contiguous()))
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        if self.use_topp_flash and not ret_attn_mask:
            if self.training:
                raise RuntimeError(
                    'topp flash optimized path is inference-only. '
                    'Set model.backbone.use_topp_flash=False for training.')
            if self.attn_vis_config.get('enabled', False):
                raise RuntimeError(
                    'topp flash inference requires attn_vis disabled.')
            if self.W and GA is not None:
                raise RuntimeError(
                    'topp flash inference does not support GA route input.')
            if not is_topp_flash_available(self.topp_flash_backend):
                raise RuntimeError(
                    'topp flash extension is unavailable.')

            route_with_cuda = str(self.topp_flash_backend or '').strip().lower() in (
                'cuda', 'cuda_forward')
            if route_with_cuda and can_run_topp_route_cuda(
                    q_win, self.router.topk):
                try:
                    r_weight, r_idx, keep_len = run_stage(
                        'router',
                        lambda: topp_route_cuda(
                            query=q_win,
                            topk=self.router.topk,
                            p=self.router.P,
                            temperature=self.router.Temperature,
                            energy=self.router.energy,
                            scale=self.router.scale,
                            debug=self.topp_flash_debug))
                    r_mask = None
                except Exception as exc:
                    warn_topp_route_cuda_fallback(str(exc))
                    r_weight, r_idx, r_mask = run_stage(
                        'router',
                        lambda: self.router(q_win, k_win, GA))
                    keep_len = r_mask.sum(dim=-1).contiguous().long()
            else:
                if route_with_cuda:
                    warn_topp_route_cuda_fallback(
                        'shape, dtype, or runtime state rejected CUDA route kernel.')
                r_weight, r_idx, r_mask = run_stage(
                    'router',
                    lambda: self.router(q_win, k_win, GA))
                keep_len = r_mask.sum(dim=-1).contiguous().long()

            out = run_stage(
                'attn',
                lambda: topp_flash_attention(
                    q_pix=q_pix,
                    kv_pix=kv_pix,
                    r_weight=r_weight,
                    r_idx=r_idx,
                    r_mask=r_mask,
                    keep_len=keep_len,
                    num_heads=self.num_heads,
                    qk_dim=self.qk_dim,
                    dim=self.dim,
                    scale=self.scale,
                    n_win=self.n_win,
                    H=H,
                    W=W,
                    backend=self.topp_flash_backend,
                    debug=self.topp_flash_debug))
            out = out + lepe
            out = run_stage('wo', lambda: self.wo(out))
            if self.auto_pad and (pad_r > 0 or pad_b > 0):
                out = out[:, :H_in, :W_in, :].contiguous()
            _log_topp_stage_debug(
                f'topp_flash_{self.topp_flash_backend or "torch"}',
                x, q_pix, kv_pix, r_idx,
                stage_times, self.num_heads, self.qk_dim, self.dim,
                self.n_win)
            return out

        # 路由机制
        r_weight, r_idx, r_mask = run_stage(
            'router',
            lambda: self.router(q_win, k_win,GA))  # all are (n, p^2, topk) tensors

        if self.use_pruned_kv_gather and not ret_attn_mask:
            out = self._attention_with_pruned_kv_gather(
                q_pix=q_pix,
                kv_pix=kv_pix,
                r_weight=r_weight,
                r_idx=r_idx,
                r_mask=r_mask,
                H=H,
                W=W)
            out = out + lepe
            out = self.wo(out)
            if self.auto_pad and (pad_r > 0 or pad_b > 0):
                out = out[:, :H_in, :W_in, :].contiguous()
            return out

        if self.use_fast_attention and not ret_attn_mask:
            return self._attention_fast(
                q_pix=q_pix, kv_pix=kv_pix,
                r_weight=r_weight, r_idx=r_idx, r_mask=r_mask,
                H=H, W=W, lepe=lepe, H_in=H_in, W_in=W_in,
                pad_r=pad_r, pad_b=pad_b)

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention    —— 注意力计算
        attn_weight = (q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        route_mask = r_mask[..., None].expand(-1, -1, -1, kv_pix_sel.size(-2))
        route_mask = rearrange(route_mask, 'n p2 k w2 -> (n p2) 1 1 (k w2)')
        attn_weight = attn_weight.masked_fill(
            ~route_mask, torch.finfo(attn_weight.dtype).min)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out

    def _attention_with_pruned_kv_gather(self, q_pix: Tensor, kv_pix: Tensor,
                                         r_weight: Tensor, r_idx: Tensor,
                                         r_mask: Tensor, H: int,
                                         W: int) -> Tensor:
        n, p2, q_len, _ = q_pix.shape
        _, _, kv_len, c_kv = kv_pix.shape
        flat_size = n * p2
        topk = r_idx.size(-1)
        head_q = self.qk_dim // self.num_heads
        head_v = self.dim // self.num_heads
        num_groups = max(1, min(int(self.pruned_kv_num_groups), topk))

        keep_len = r_mask.sum(dim=-1).reshape(flat_size).long().clamp(min=1)
        q_flat = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                           m=self.num_heads)
        idx_flat = r_idx.reshape(flat_size, topk).long()
        weight_flat = r_weight.reshape(flat_size, topk).to(kv_pix.dtype)
        out_flat = q_flat.new_empty(flat_size, self.num_heads, q_len, head_v)

        sorted_keep, sorted_idx = torch.sort(keep_len)
        boundaries = torch.linspace(0, 1, num_groups + 1,
                                    device=keep_len.device)
        quantiles = torch.quantile(keep_len.float(), boundaries).long().clamp(
            min=1, max=topk)
        quantiles[-1] = topk
        all_bounds = torch.stack([quantiles[:-1], quantiles[1:] + 1]).T
        edges = torch.searchsorted(sorted_keep, all_bounds.flatten())
        edges_cpu = edges.cpu().tolist()

        for g in range(num_groups):
            left = edges_cpu[2 * g]
            right = edges_cpu[2 * g + 1]
            flat_ids = sorted_idx[left:right]
            if flat_ids.numel() == 0:
                continue

            group_keep = keep_len.index_select(0, flat_ids)
            group_max_keep = int(group_keep.max().item())
            n_ids = torch.div(flat_ids, p2, rounding_mode='floor')
            kept_b_idx = idx_flat.index_select(0, flat_ids)[:, :group_max_keep]
            q = q_flat.index_select(0, flat_ids)
            kv_batch = kv_pix.index_select(0, n_ids)
            group_col = torch.arange(group_max_keep, device=q_pix.device)

            kv_sel = torch.gather(
                kv_batch,
                dim=1,
                index=kept_b_idx[:, :, None, None].expand(-1, -1, kv_len,
                                                          c_kv))
            weight = weight_flat.index_select(0, flat_ids)[:, :group_max_keep]
            kv_sel = weight[:, :, None, None] * kv_sel
            k_sel, v_sel = kv_sel.split([self.qk_dim, self.dim], dim=-1)

            k_sel = k_sel.view(-1, group_max_keep, kv_len, self.num_heads,
                               head_q).permute(0, 3, 4, 1, 2)
            v_sel = v_sel.view(-1, group_max_keep, kv_len, self.num_heads,
                               head_v).permute(0, 3, 1, 2, 4)

            q = q * self.scale
            mask_group = torch.zeros(q.size(0), group_max_keep,
                                     device=q.device, dtype=kv_pix.dtype)
            mask_group.masked_fill_(
                group_col >= group_keep[:, None],
                float('-inf'))
            mask = mask_group[:, :, None].expand(-1, -1, kv_len).reshape(
                q.size(0), -1).unsqueeze(1).unsqueeze(1)
            attn = self.attn_act(q @ k_sel.flatten(3) + mask)
            out_chunk = attn @ v_sel.reshape(
                -1, self.num_heads, group_max_keep * kv_len, head_v)

            out_flat.index_copy_(0, flat_ids, out_chunk)

        return rearrange(
            out_flat,
            '(n j i) m (h w) c -> n (j h) (i w) (m c)',
            n=n,
            j=self.n_win,
            i=self.n_win,
            h=H // self.n_win,
            w=W // self.n_win)

    def _attention_fast(self, q_pix, kv_pix, r_weight, r_idx, r_mask,
                        H, W, lepe, H_in, W_in, pad_r, pad_b):
        keep_len = r_mask.sum(dim=-1)
        keep_max = int(keep_len.max().clamp(min=1).item())
        topk = r_idx.size(-1)
        keep_max = min(keep_max, topk)

        r_idx_f = r_idx[:, :, :keep_max]
        r_weight_f = r_weight[:, :, :keep_max]
        r_mask_f = r_mask[:, :, :keep_max]

        kv_pix_sel = self.kv_gather(r_idx=r_idx_f, r_weight=r_weight_f, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        route_mask = r_mask_f[..., None].expand(-1, -1, -1, kv_pix.size(-2))
        route_mask = rearrange(route_mask, 'n p2 k w2 -> (n p2) 1 1 (k w2)')
        attn_weight = attn_weight.masked_fill(
            ~route_mask, torch.finfo(attn_weight.dtype).min)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)',
                        j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        out = self.wo(out)
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
        return out
