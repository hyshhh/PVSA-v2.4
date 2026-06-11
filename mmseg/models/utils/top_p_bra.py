from typing import Tuple
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .topp_flash_kernel import is_topp_flash_available, topp_flash_attention

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

    def __init__(self,qk_dim,topk=4, qk_scale=None, param_routing=False, diff_routing=False,W=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.W=W
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activate
        self.routing_act = nn.Softmax(dim=-1)
        self.flag=0
#top—p-v2\3_2025_12_25,关于top-p和温度的调整，v3.0添加能量补偿因子
        if self.topk==16:
            self.topk=25
            self.P= 0.2  
            self.Temperature=0.0175
            self.energy=4
        elif self.topk==12:
            self.topk=18
            self.P = 0.4  
            self.Temperature=0.025
            self.energy=1.5
        elif self.topk==8:
            self.topk=36
            self.P = 0.6
            self.Temperature=0.05  
            self.energy=0.75
        elif self.topk==6:
            self.P = 0.8
            self.topk=49
            self.Temperature=0.15
            self.energy=0.4
        
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
        attn[0][32] = -torch.tensor([
            0.127, 0.120, 0.104, 0.125, 0.112, 0.126, 0.108, 0.125,
            0.131, 0.110, 0.107, 0.109, 0.100, 0.102, 0.123, 0.090,
            0.125, 0.096, 0.094, 0.082, 0.096, 0.095, 0.042, 0.092,
            0.116, 0.078, 0.104, 0.005, -0.098, -0.088, -0.102, -0.075,
            -0.045, -0.106, -0.088, -0.117, -0.091, -0.088, -0.081, -0.093,
            -0.120, -0.122, -0.097, -0.122, -0.128, -0.124, -0.122, -0.124,
            -0.128
            ], device='cuda:0')
        # 3️⃣ Top-k selection (no sorting for speed)
        topk_score, topk_index = torch.topk(
            attn, k=self.topk, dim=-1, sorted=True
        )  # (n, p2, k)
        if self.flag==0 and self.topk==9:
            print("第0",attn[0][2])
            
            overlay = overlay_attn_block("/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/data/gqyyz/image/test1/2024_08_19_195748172_cropped_1.jpg", attn[0][32])
            cv2.imwrite("/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/cam/attn/attn_stage.png", overlay)

        topk_score = torch.softmax(topk_score /self.Temperature, dim=-1)
        if self.flag==0 and self.topk==25:
            img_path = "/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/data/gqyyz/image/test1/2024_08_19_195748172_cropped_1.jpg"
            
            total_patches = attn.shape[-1] 
            overlay_topk = overlay_topk_attn_block_no_heatmap(
                img=img_path, 
                topk_score=topk_score[0][32], 
                topk_index=topk_index[0][32], 
                total_patches=total_patches,
                dark_ratio=0.3 # 背景变暗比例，0.3表示背景保留30%亮度，你可以自己调！
            )
            save_path = "/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/cam/attn/attn_stage_topk.png"
            cv2.imwrite(save_path, overlay_topk)
            # print("第1",topk_score[0][0])
            # print("第1'",topk_index[0][0])

        # 5️⃣ Cumulative probability pruning
        cumsum = torch.cumsum(topk_score, dim=-1)      # (n, p2, k)

        keep_mask = cumsum <= self.P                   # (n, p2, k)
        keep_len = keep_mask.sum(dim=-1, keepdim=True) # (n, p2, 1)
        keep_len = keep_len.clamp(min=1)

        # 6️⃣ Vectorized truncation (NO LOOP)
        max_len = keep_len.max()

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
        topk_score = topk_score * max_len* self.energy

        if self.flag==4 and self.topk==4:
            print("第3",topk_score[0][0])
            print("第3",topk_index[0][0])

        # 7️⃣ Final routing activation
        # r_weight = self.routing_act(topk_score)
        # if self.flag==0:
        #     print("第4",r_weight[0][0])

        return topk_score, topk_index, valid_mask

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
                 topp_flash_block_windows=64, topp_flash_backend=None):
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
                                  param_routing=self.param_routing,W=self.W)
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

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # 对值部分应用深度可分离卷积作为位置编码
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        # 路由机制
        r_weight, r_idx, r_mask = self.router(q_win, k_win,GA)  # all are (n, p^2, topk) tensors

        if self.use_topp_flash and not ret_attn_mask and is_topp_flash_available(
                self.topp_flash_backend):
            out = topp_flash_attention(
                q_pix=q_pix,
                kv_pix=kv_pix,
                r_weight=r_weight,
                r_idx=r_idx,
                r_mask=r_mask,
                num_heads=self.num_heads,
                qk_dim=self.qk_dim,
                dim=self.dim,
                scale=self.scale,
                n_win=self.n_win,
                H=H,
                W=W,
                block_windows=self.topp_flash_block_windows,
                backend=self.topp_flash_backend)
            out = out + lepe
            out = self.wo(out)
            if self.auto_pad and (pad_r > 0 or pad_b > 0):
                out = out[:, :H_in, :W_in, :].contiguous()
            return out

        if self.use_topp_flash and not self._topp_flash_warned:
            warnings.warn(
                'topp flash attention kernel is unavailable; '
                'fallback to the torch implementation.')
            self._topp_flash_warned = True

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
