import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from ..utils.common import Attention, AttentionLePE, DWConv
# from ..utils.bra_legacy import BiLevelRoutingAttention
from ..utils.bra_legacy_hys_v4 import BiLevelRoutingAttention
from ..utils import top_p_bra as _tpb
from ..utils.top_p_bra import ToppAttention
from mmseg.registry import MODELS


def _normalize_topp_backend(backend):
    if backend is None:
        return None
    backend = str(backend).strip().lower()
    if backend in ('', 'none', 'false', 'off'):
        return None
    return backend


def _fuse_conv_bn(conv, bn):
    if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
        return conv, bn
    if conv.training or bn.training:
        return conv, bn
    fused = fuse_conv_bn_eval(conv, bn)
    return fused, nn.Identity()


def _fuse_sequential_conv_bn(module):
    for child in module.children():
        _fuse_sequential_conv_bn(child)
    if not isinstance(module, nn.Sequential):
        return
    children = list(module.children())
    i = 0
    while i + 1 < len(children):
        if isinstance(children[i], nn.Conv2d) and isinstance(
                children[i + 1], nn.modules.batchnorm._BatchNorm):
            children[i], children[i + 1] = _fuse_conv_bn(
                children[i], children[i + 1])
            i += 2
        else:
            i += 1
    module._modules.clear()
    for idx, child in enumerate(children):
        module.add_module(str(idx), child)



def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False, W=False,
                 topp_flash_block_windows=64,
                 topp_flash_backend=None,
                 use_pruned_kv_gather=False, pruned_kv_num_groups=1,
                 topp_route_configs=None,
                 attn_vis_config=None,
                 use_fast_attention=False,
                 debug_route=False,
                 topp_flash_debug=False,
                 use_route_mask=False,
                 use_nan_guard=False,
                 use_plain_attn=False,
                 attention_type='topp'):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        self.W = W
        # 如果在注意力前加入卷积核：
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        # topk<=0（如 BRA 默认最后一阶段 -1）或显式 plain：走普通 self-attention
        if use_plain_attn or topk <= 0:
            self.PA = Attention(dim=dim, num_heads=num_heads)
            self._use_plain_attn = True
        elif attention_type == 'topp':
            self.PA = ToppAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                    qk_scale=qk_scale, kv_per_win=kv_per_win,
                                    kv_downsample_ratio=kv_downsample_ratio,
                                    kv_downsample_kernel=kv_downsample_kernel,
                                    kv_downsample_mode=kv_downsample_mode,
                                    topk=topk, param_attention=param_attention, param_routing=param_routing,
                                    diff_routing=diff_routing, soft_routing=soft_routing,
                                    side_dwconv=side_dwconv,
                                    auto_pad=auto_pad, W=self.W,
                                    topp_flash_block_windows=topp_flash_block_windows,
                                    topp_flash_backend=topp_flash_backend,
                                    use_pruned_kv_gather=use_pruned_kv_gather,
                                    pruned_kv_num_groups=pruned_kv_num_groups,
                                    topp_route_configs=topp_route_configs,
                                    attn_vis_config=attn_vis_config,
                                    use_fast_attention=use_fast_attention,
                                    debug_route=debug_route,
                                    topp_flash_debug=topp_flash_debug,
                                    use_route_mask=use_route_mask,
                                    use_nan_guard=use_nan_guard)
            self._use_plain_attn = False
        elif attention_type == 'bra':
            # BRA: 标准 Bi-Level Routing Attention（固定 top-k，无 Top-P 裁剪）
            self.PA = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                              qk_scale=qk_scale, kv_per_win=kv_per_win,
                                              kv_downsample_ratio=kv_downsample_ratio,
                                              kv_downsample_kernel=kv_downsample_kernel,
                                              kv_downsample_mode=kv_downsample_mode,
                                              topk=topk, param_attention=param_attention,
                                              param_routing=param_routing,
                                              diff_routing=diff_routing,
                                              soft_routing=soft_routing,
                                              side_dwconv=side_dwconv,
                                              auto_pad=auto_pad)
            self._use_plain_attn = False
        else:
            raise ValueError(f'Unsupported attention_type: {attention_type}')
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.norm4 = nn.LayerNorm(dim, eps=1e-6)
        # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
        #                          DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
        #                          nn.GELU(),
        #                          nn.Linear(int(mlp_ratio * dim), dim)
        #                          )
        self.mlp2 = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
        

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # VTFormerv1.22,只有Top-p
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        if self._use_plain_attn:
            PA = self.PA(self.norm3(x))
        else:
            PA = self.PA(self.norm3(x), None)
        if self.pre_norm:
                x = x + self.drop_path(PA)   # (N, H, W, C)
                x = x + self.drop_path(self.mlp2(self.norm4(x)))  # (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x


class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5,
                 use_channel=True, use_spatial=True):
        super(FeatureAlignmentModule, self).__init__()
        if not use_channel and not use_spatial:
            raise ValueError('FeatureAlignmentModule 至少需要开启 CA 或 SA 之一')
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        # sigmoid(0) = 0.5, so 2*sigmoid(0) = 1.0 — neutral init
        if use_channel:
            self.lambda_c = nn.Parameter(torch.tensor(0.0))
            self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        if use_spatial:
            self.lambda_s = nn.Parameter(torch.tensor(0.0))
            self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        out_x1, out_x2 = x1, x2
        if self.use_channel:
            channel_weights = self.channel_weights(x1, x2)
            lc = 2.0 * self.lambda_c.sigmoid()
            out_x1 = out_x1 + lc * channel_weights[1] * x2
            out_x2 = out_x2 + lc * channel_weights[0] * x1
        if self.use_spatial:
            spatial_weights = self.spatial_weights(x1, x2)
            ls = 2.0 * self.lambda_s.sigmoid()
            out_x1 = out_x1 + ls * spatial_weights[1] * x2
            out_x2 = out_x2 + ls * spatial_weights[0] * x1
        return out_x1, out_x2
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
import torch
import torch.nn as nn

class DepthWiseConvModule(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 drop_rate=0.,
                 dilation=1,
                 activate_after_dw=False):
        super(DepthWiseConvModule, self).__init__()
        self.activate_after_dw = activate_after_dw

        # 1. 自动计算 Padding，保证 stride=1 时尺寸不变
        # 考虑到 dilation 的情况: padding = dilation * (kernel_size - 1) // 2
        padding = dilation * (kernel_size - 1) // 2

        # 2. 第一个点卷积 (1x1 Conv): 升维 (Expansion)
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feedforward_channels) # 加上 BN
        # 3. 深度卷积 (Depthwise Conv)
        self.pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=feedforward_channels, # 关键：Groups = Channels
            bias=False)
        self.bn2 = nn.BatchNorm2d(feedforward_channels) # 加上 BN
        self.activate = nn.GELU() # 或者是 build_activation_layer(act_cfg)
        # 4. 第二个点卷积 (1x1 Conv): 降维 (Projection)
        self.fc2 = nn.Conv2d(feedforward_channels, output_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels) # 加上 BN
        self.drop = nn.Dropout(drop_rate)
        # 处理残差连接时的维度/步长不匹配问题
        self.downsample = None
        if stride != 1 or embed_dims != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(embed_dims, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
    def forward(self, x):
        identity = x

        # 典型的结构：Conv -> BN -> Act -> Conv -> BN -> Act ...
        # 这里采用类似 MobileNetV2/SegFormer 的顺序
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activate(out) # 升维后激活
        out = self.pe_conv(out)
        out = self.bn2(out)
        if self.activate_after_dw:
            out = self.activate(out)
        out = self.fc2(out)
        out = self.bn3(out)
        # 最后通常不激活，直接做 Dropout 和 Add
        out = self.drop(out)
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

    def fuse_for_inference(self):
        if self.training:
            return
        self.fc1, self.bn1 = _fuse_conv_bn(self.fc1, self.bn1)
        self.pe_conv, self.bn2 = _fuse_conv_bn(self.pe_conv, self.bn2)
        self.fc2, self.bn3 = _fuse_conv_bn(self.fc2, self.bn3)
        if self.downsample is not None:
            _fuse_sequential_conv_bn(self.downsample)

class MBConv(nn.Module):
    """EfficientNet MBConv 块：升维→DWConv→SE→降维，SiLU 激活"""
    def __init__(self, embed_dims, feedforward_channels, output_channels,
                 kernel_size=3, stride=1, se_ratio=0.25, drop_rate=0.,
                 use_se=True):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = (stride == 1 and embed_dims == output_channels)
        self.use_se = use_se
        # 升维
        self.expand = nn.Sequential(
            nn.Conv2d(embed_dims, feedforward_channels, 1, bias=False),
            nn.BatchNorm2d(feedforward_channels),
            nn.SiLU())
        # 深度卷积
        self.dw_conv = nn.Sequential(
            nn.Conv2d(feedforward_channels, feedforward_channels,
                      kernel_size, stride, padding,
                      groups=feedforward_channels, bias=False),
            nn.BatchNorm2d(feedforward_channels),
            nn.SiLU())
        # SE 注意力
        if self.use_se:
            se_channels = max(1, int(embed_dims * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feedforward_channels, se_channels, 1),
                nn.SiLU(),
                nn.Conv2d(se_channels, feedforward_channels, 1),
                nn.Sigmoid())
        # 降维
        self.proj = nn.Sequential(
            nn.Conv2d(feedforward_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels))
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.dw_conv(x)
        if self.use_se:
            x = x * self.se(x)
        x = self.proj(x)
        x = self.drop(x)
        if self.use_residual:
            x = x + residual
        return x

    def fuse_for_inference(self):
        if self.training:
            return
        self.expand[0], self.expand[1] = _fuse_conv_bn(self.expand[0], self.expand[1])
        self.dw_conv[0], self.dw_conv[1] = _fuse_conv_bn(self.dw_conv[0], self.dw_conv[1])
        self.proj[0], self.proj[1] = _fuse_conv_bn(self.proj[0], self.proj[1])


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_for_inference(self):
        if self.training:
            return
        self.conv, self.bn = _fuse_conv_bn(self.conv, self.bn)


class C2fBottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.cv2 = ConvBNAct(channels, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out

    def fuse_for_inference(self):
        self.cv1.fuse_for_inference()
        self.cv2.fuse_for_inference()


class C2fBlock(nn.Module):
    """YOLO 风格 C2f 块，用于验证跨阶段部分连接的 CNN 分支收益。"""
    def __init__(self, channels, hidden_ratio=0.5, num_blocks=2):
        super().__init__()
        hidden_channels = max(1, int(channels * hidden_ratio))
        self.cv1 = ConvBNAct(channels, 2 * hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            C2fBottleneck(hidden_channels) for _ in range(num_blocks)
        ])
        self.cv2 = ConvBNAct(
            (2 + num_blocks) * hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.cv2(torch.cat(y, dim=1))

    def fuse_for_inference(self):
        self.cv1.fuse_for_inference()
        for block in self.blocks:
            block.fuse_for_inference()
        self.cv2.fuse_for_inference()


class C3k2Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.cv2 = ConvBNAct(channels, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out

    def fuse_for_inference(self):
        self.cv1.fuse_for_inference()
        self.cv2.fuse_for_inference()


class C3k2Block(nn.Module):
    """C3k2 风格块，用于对比更强卷积分支的局部建模能力。"""
    def __init__(self, channels, hidden_ratio=0.5, num_blocks=2):
        super().__init__()
        hidden_channels = max(1, int(channels * hidden_ratio))
        self.cv1 = ConvBNAct(channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNAct(channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            C3k2Bottleneck(hidden_channels) for _ in range(num_blocks)
        ])
        self.cv3 = ConvBNAct(2 * hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))

    def fuse_for_inference(self):
        self.cv1.fuse_for_inference()
        self.cv2.fuse_for_inference()
        for block in self.blocks:
            block.fuse_for_inference()
        self.cv3.fuse_for_inference()


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块，用于验证大核深度卷积和通道 MLP 的收益。"""
    def __init__(self, channels, layer_scale=1e-6, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3,
            groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels))
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.gamma.view(1, 1, 1, -1) * out
        out = out.permute(0, 3, 1, 2)
        return residual + self.drop(out)


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，(B, 96, 256, 256) → (B, 96, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(
                    nn.Linear(self.dim, self.dim),#如果我的输入向量是96，但是全连接层在
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp_max = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # print("!!!!!!!!!!!!")
        # print(B, C, H, W)#(1,12,256,256)
        x = torch.cat((x1, x2), dim=1)
        # print("a")
        # print(x.shape)

        # Avg. Adaptive normalization
        avg = self.avg_pool(x).view(B, 2 * C)
        # print("b")
        # print("avg shape:", avg.shape)
        avg_attn = self.mlp_avg(avg).softmax(dim=-1)
        avg_x1, avg_x2 = (avg_attn.view(B, 2, 1) * avg.view(B, 2, C)).chunk(2, dim=1)
        avg_x = (avg_x1 + avg_x2).view(B, C)

        # Max. Adaptive normalization
        max = self.max_pool(x).view(B, 2 * C)
        max_attn = self.mlp_max(max).softmax(dim=-1)
        max_x1, max_x2 = (max_attn.view(B, 2, 1) * max.view(B, 2, C)).chunk(2, dim=1)
        max_x = (max_x1 + max_x2).view(B, C)

        y = torch.cat((avg_x, max_x), dim=1)
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weights = y.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
class Stem224(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128):
        super().__init__()
        self.conv1 = DepthWiseConvModule(
            in_chans, embed_dim // 2, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.act1 = nn.GELU()

        self.conv2 = DepthWiseConvModule(
            embed_dim // 2, embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.act2 = nn.GELU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 1, stride=4),  # 下采样 + 对齐通道
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out
@MODELS.register_module()
class VTFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 norm_eval=False,
                 W=False,
                 topp_flash_backend=None,
                 topp_flash_block_windows=64,
                 topp_flash_debug=False,
                 # CUDA inference params
                 use_pruned_kv_gather=False,
                 pruned_kv_num_groups=1,
                 topp_route_configs=None,
                 attn_vis_config=None,
                 use_fast_attention=False,
                 debug_route=False,
                 use_route_mask=False,
                 use_nan_guard=False,
                 fam_reduction=4,
                 cnn_block_layers=[2, 1, 2, 1],
                 cnn_block_type='dwconv',
                 feature_vis_config=None,
                 use_fam=True,
                 fam_use_channel=True,
                 fam_use_spatial=True,
                 use_plain_attn_last_stage=False,
                 attention_type='topp',
                 route_pooling='avg',
                 **kwargs):

        super().__init__()
        self.W = W
        self.topp_flash_backend = _normalize_topp_backend(topp_flash_backend)
        self.topp_flash_block_windows = topp_flash_block_windows
        self.topp_flash_debug = topp_flash_debug
        self.use_pruned_kv_gather = use_pruned_kv_gather
        self.pruned_kv_num_groups = pruned_kv_num_groups
        self.topp_route_configs = topp_route_configs
        self.attn_vis_config = attn_vis_config
        self.use_fast_attention = use_fast_attention
        self.debug_route = debug_route
        self.use_route_mask = use_route_mask
        self.use_nan_guard = use_nan_guard
        self.feature_vis_config = feature_vis_config or {}
        self._inference_fused = False
        self._disable_inference_fusion = False
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.norm_eval = norm_eval
        self.use_fam = use_fam
        self.fam_use_channel = fam_use_channel
        self.fam_use_spatial = fam_use_spatial
        # CA/SA 全关时等价于关闭 FAM，避免空模块仍被统计
        if use_fam and (not fam_use_channel) and (not fam_use_spatial):
            self.use_fam = False
        self.attention_type = attention_type
        self.use_plain_attn_last_stage = use_plain_attn_last_stage
        self.route_pooling = route_pooling
        # BRA 模式固定使用标准 BiFormer topks，避免沿用 PVSA 配置里的 [16,12,8,6]。
        # PVSA(topp) 才使用配置传入的 topks；未传时回退 [8,8,-1,-1]。
        if attention_type == 'bra':
            topks = [1, 4, 16, -1]
        elif topks is None:
            topks = [8, 8, -1, -1]
        self.topks = list(topks)
        # cnn_block_layers 全零时禁用 CNN 分支，只走 Transformer
        self._cnn_disabled = all(v == 0 for v in cnn_block_layers)
        ############ downsample layers (patch embeddings) ######################
        # CNN block 工厂函数
        valid_cnn_block_types = {
            'dwconv', 'dwconv_act', 'mbconv', 'mbconv_no_se',
            'c2f', 'c3k2', 'convnext'
        }
        if cnn_block_type not in valid_cnn_block_types:
            raise ValueError(
                f'cnn_block_type must be one of {valid_cnn_block_types}, '
                f'but got {cnn_block_type}')
        self.cnn_block_type = cnn_block_type
        expansion = 4
        def _make_cnn_block(ch):
            if cnn_block_type == 'mbconv':
                return MBConv(ch, expansion * ch, ch)
            if cnn_block_type == 'mbconv_no_se':
                return MBConv(ch, expansion * ch, ch, use_se=False)
            if cnn_block_type == 'c2f':
                return C2fBlock(ch)
            if cnn_block_type == 'c3k2':
                return C3k2Block(ch)
            if cnn_block_type == 'convnext':
                return ConvNeXtBlock(ch)
            return DepthWiseConvModule(
                ch, expansion * ch, ch,
                activate_after_dw=(cnn_block_type == 'dwconv_act'))

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers2 = nn.ModuleList()
        self.FAM = nn.ModuleList()
        # cnn_block_layers 全 0 时，整条 CNN 分支 + 同层 fusion 都不创建
        if self._cnn_disabled:
            self.use_fam = False



        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        stem2_layers = [
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        ]
        stem2_layers.extend([
            _make_cnn_block(embed_dim[0])
            for _ in range(cnn_block_layers[0])
        ])
        stem2 = nn.Sequential(*stem2_layers)

        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
            stem2.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
            stem2 = checkpoint_wrapper(stem2)
        self.downsample_layers.append(stem)
        self.fusion = nn.ModuleList()
        fusion_builder = getattr(
            self, '_build_fusion_layer',
            lambda channels: nn.Conv2d(
                2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=True))
        # cnn_block_layers 全 0 时，整条 CNN 分支 + 同层 fusion 都不创建
        if not self._cnn_disabled:
            self.downsample_layers2.append(stem2)
            if self.use_fam:
                self.FAM.append(FeatureAlignmentModule(
                    dim=2 * embed_dim[0], reduction=fam_reduction,
                    use_channel=fam_use_channel, use_spatial=fam_use_spatial))
            self.fusion.append(fusion_builder(embed_dim[0]))
        else:
            # 纯 Transformer 模式占位，保持索引对齐
            self.downsample_layers2.append(nn.Identity())
            self.fusion.append(nn.Identity())

        self.norm = nn.LayerNorm(normalized_shape=1)  # 根据实际维度调整
        # 定义Sigmoid激活
        self.sigmoid = nn.Sigmoid()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            layers = [
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            ]
            layers.extend([
            _make_cnn_block(embed_dim[i + 1])
            for _ in range(cnn_block_layers[i + 1])
             ])
            downsample_layer2 = nn.Sequential(*layers)
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
                downsample_layer2.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
                downsample_layer2 = checkpoint_wrapper(downsample_layer2)
            self.downsample_layers.append(downsample_layer)
            if not self._cnn_disabled:
                self.downsample_layers2.append(downsample_layer2)
                self.fusion.append(fusion_builder(embed_dim[i + 1]))
                if self.use_fam:
                    self.FAM.append(FeatureAlignmentModule(
                        dim=2 * embed_dim[i + 1], reduction=fam_reduction,
                        use_channel=fam_use_channel, use_spatial=fam_use_spatial))
            else:
                self.downsample_layers2.append(nn.Identity())
                self.fusion.append(nn.Identity())

        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=self.topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        W=self.W,
                        topp_flash_block_windows=self.topp_flash_block_windows,
                        topp_flash_backend=self.topp_flash_backend,
                        use_pruned_kv_gather=self.use_pruned_kv_gather,
                        pruned_kv_num_groups=self.pruned_kv_num_groups,
                        topp_route_configs=self.topp_route_configs,
                        attn_vis_config=self.attn_vis_config,
                        use_fast_attention=self.use_fast_attention,
                        debug_route=self.debug_route,
                        topp_flash_debug=self.topp_flash_debug,
                        use_route_mask=self.use_route_mask,
                        use_nan_guard=self.use_nan_guard,
                        use_plain_attn=(
                            (self.use_plain_attn_last_stage and i == 3) or self.topks[i] <= 0),
                        attention_type=self.attention_type
                        ) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def optimize_for_inference(self):
        if (self.training or self._inference_fused
                or self._disable_inference_fusion):
            return
        for layer in self.downsample_layers:
            _fuse_sequential_conv_bn(layer)
        for layer in self.downsample_layers2:
            _fuse_sequential_conv_bn(layer)
        for module in self.modules():
            if isinstance(module, (DepthWiseConvModule, MBConv, ConvBNAct,
                                   C2fBottleneck, C2fBlock, C3k2Bottleneck,
                                   C3k2Block, ConvNeXtBlock)):
                module.fuse_for_inference()
        self._inference_fused = True

    def forward_features(self, x):
        # 图级 debug 计时开关：由 backbone.topp_flash_debug 控制，
        # 与 ToppAttention 内采样共用；每图末尾统一结算（每 ROUND 张打印一次）。
        _tpb._STAGE_DEBUG_ACTIVE = _tpb._normalize_flash_debug(
            self.topp_flash_debug)
        _tpb._STAGE_BLOCK_INDEX = 0
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            if i == 3 and _tpb._STAGE_DEBUG_ACTIVE > 0:
                # S4 是 plain Attention（不走 ToppAttention），单独计时。
                # 模式2：逐个 block 计时并立即打印；模式1：整个 stage 累计到图级结算。
                _dim4 = self.embed_dim[3]
                _dbg = _tpb._STAGE_DEBUG_ACTIVE
                if _dbg >= 2:
                    # 模式2：逐 block 整块单次计时，立即打印（与 S1..S3 口径一致）
                    _x4 = x
                    for _bi, _blk in enumerate(self.stages[i]):
                        _tpb._STAGE_BLOCK_INDEX += 1
                        with torch.no_grad():
                            _xo, _e = _tpb._time_cuda_stage(
                                _dbg, _x4, lambda b=_blk: b(_x4))
                        if _e is not None:
                            _tpb._log_topp_stage_debug(
                                stage=f'S4[{_tpb._STAGE_BLOCK_INDEX}]',
                                path='plain_attention',
                                x=tuple(_x4.shape), q_pix=tuple(_x4.shape),
                                kv_pix=tuple(_x4.shape), r_idx=(),
                                times={'BLOCK': _e}, num_heads=0,
                                qk_dim=_dim4, dim=_dim4, n_win=0)
                        _x4 = _xo
                    x = _x4
                else:
                    # 模式1：逐 block 计时，累计后按 block 数求平均，
                    # 与 S1..S3 的单 block 平均口径一致。
                    _x4 = x
                    _acc4 = 0.0
                    _cnt4 = 0
                    for _blk in self.stages[i]:
                        with torch.no_grad():
                            _xo, _e = _tpb._time_cuda_stage(
                                _dbg, _x4, lambda b=_blk: b(_x4))
                        if _e is not None:
                            _acc4 += _e
                            _cnt4 += 1
                        _x4 = _xo
                    if _cnt4 > 0:
                        _tpb.add_stage_round_entry(
                            _dim4, 'attn', _acc4 / _cnt4)
                        _tpb._STAGE_ROUND_INFO.setdefault(_dim4, dict(
                            path='plain_attention',
                            x_shape=tuple(x.shape),
                            q_shape=tuple(x.shape),
                            kv_shape=tuple(x.shape),
                            route_shape=(),
                            num_heads=0,
                            qk_dim=_dim4,
                            dim=_dim4,
                            n_win=0))
                    x = _x4
            else:
                x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        # 每图末尾统一结算：满 ROUND 张图打印 S1..S4 各一行
        _tpb._finalize_stage_round()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        return x

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

#################### model variants #######################
