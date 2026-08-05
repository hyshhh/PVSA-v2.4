"""对比实验中的 BiFormer 主干。

这里使用标准 Bi-Level Routing Attention（前三个阶段）和末阶段全局注意力，
不修改原有 PVSA-Net 文件，专门用于速度与阶段耗时对比。
"""

import torch
import torch.nn as nn

from mmseg.registry import MODELS

from .attention_timing import (CompareAttentionBlock, CompareBackboneBase,
                                window_partition, window_reverse)


class GlobalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("注意力通道数必须能被头数整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        tokens = height * width
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads,
                                  self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj(output)
        return output.reshape(batch, height, width, channels)


class BiLevelRoutingAttention(nn.Module):
    """简洁、可变输入尺寸的 Bi-Level Routing Attention。"""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 topk: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("注意力通道数必须能被头数整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = int(window_size)
        self.topk = int(topk)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.lepe = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        window_size = max(1, min(self.window_size, height, width))
        windows, pad_hw = window_partition(x, window_size)
        padded_height = height + pad_hw[0]
        padded_width = width + pad_hw[1]
        windows_per_image = (padded_height // window_size) * (
            padded_width // window_size)
        tokens_per_window = window_size * window_size

        qkv = self.qkv(windows).reshape(
            batch, windows_per_image, tokens_per_window, 3,
            self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=3)
        # [B, Nw, T, heads, head_dim]
        route_query = query.reshape(batch, windows_per_image,
                                    tokens_per_window, channels).mean(dim=2)
        route_key = key.reshape(batch, windows_per_image,
                                tokens_per_window, channels).mean(dim=2)
        route_score = torch.matmul(route_query,
                                   route_key.transpose(-1, -2))
        route_score = route_score * (channels**-0.5)
        route_k = min(max(self.topk, 1), windows_per_image)
        route_index = route_score.topk(route_k, dim=-1).indices

        query = query.permute(0, 1, 3, 2, 4).contiguous()
        key = key.permute(0, 1, 3, 2, 4).contiguous()
        value = value.permute(0, 1, 3, 2, 4).contiguous()
        # 通过沿窗口维 gather 取得每个查询窗口的候选键和值。
        key_source = key.unsqueeze(1).expand(
            batch, windows_per_image, windows_per_image,
            self.num_heads, tokens_per_window, self.head_dim)
        value_source = value.unsqueeze(1).expand_as(key_source)
        index = route_index[:, :, :, None, None, None].expand(
            batch, windows_per_image, route_k, self.num_heads,
            tokens_per_window, self.head_dim)
        selected_key = torch.gather(key_source, 2, index)
        selected_value = torch.gather(value_source, 2, index)

        query = query.unsqueeze(2).expand(
            batch, windows_per_image, route_k, self.num_heads,
            tokens_per_window, self.head_dim)
        attention = torch.einsum(
            'bnkhqd,bnkhtd->bnkhqt', query, selected_key) * self.scale
        attention = attention.reshape(
            batch, windows_per_image, self.num_heads, tokens_per_window,
            route_k * tokens_per_window).softmax(dim=-1)
        selected_value = selected_value.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, windows_per_image, self.num_heads,
            route_k * tokens_per_window, self.head_dim)
        output = torch.matmul(attention, selected_value)
        output = output.permute(0, 1, 3, 2, 4).reshape(
            batch * windows_per_image, tokens_per_window, channels)

        # 局部位置编码与标准 BiFormer 的局部增强保持一致。
        value_windows = value.permute(0, 1, 3, 2, 4).reshape(
            batch * windows_per_image, tokens_per_window, channels)
        value_image = window_reverse(value_windows, window_size, batch,
                                     height, width, pad_hw)
        local = self.lepe(value_image.permute(0, 3, 1, 2))
        local = local.permute(0, 2, 3, 1)
        local_windows, _ = window_partition(local, window_size)
        output = output + local_windows
        output = self.proj(output)
        output = window_reverse(output, window_size, batch, height, width,
                                pad_hw)
        return output


_BIFORMER_VARIANTS = {
    "tiny": dict(
        embed_dims=(64, 128, 256, 512),
        depths=(2, 2, 8, 2),
        num_heads=(2, 4, 8, 16)),
    "small": dict(
        embed_dims=(64, 128, 320, 512),
        depths=(3, 4, 8, 3),
        num_heads=(2, 4, 8, 16)),
    "base": dict(
        embed_dims=(96, 192, 384, 768),
        depths=(4, 4, 12, 4),
        num_heads=(3, 6, 12, 24)),
}


@MODELS.register_module()
class CompareBiFormer(CompareBackboneBase):
    """BiFormer-T、BiFormer-S、BiFormer-B 的统一实现。"""

    def __init__(self,
                 variant: str = "tiny",
                 window_size: int = 7,
                 topks=(1, 4, 16, 1),
                 mlp_ratio: float = 4.0,
                 **kwargs) -> None:
        variant = str(variant).lower()
        aliases = {"t": "tiny", "s": "small", "b": "base"}
        variant = aliases.get(variant, variant)
        if variant not in _BIFORMER_VARIANTS:
            raise KeyError(
                f"未知 BiFormer 版本 {variant}，可选值为 {tuple(_BIFORMER_VARIANTS)}")
        spec = dict(_BIFORMER_VARIANTS[variant])
        embed_dims = kwargs.pop("embed_dims", spec["embed_dims"])
        depths = kwargs.pop("depths", spec["depths"])
        num_heads = kwargs.pop("num_heads", spec["num_heads"])
        self.variant = variant
        self.window_size = int(window_size)
        self.topks = tuple(topks)
        model_name = kwargs.pop("model_name", f"biformer_{variant}")
        in_chans = kwargs.pop("in_chans", 3)
        super().__init__(
            in_chans=in_chans,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            model_name=model_name,
            **kwargs)

    def _make_block(self, stage_idx: int, block_idx: int,
                    drop_path: float, mlp_ratio: float) -> nn.Module:
        if stage_idx == 3:
            attention = GlobalSelfAttention(self.embed_dims[stage_idx],
                                            self.num_heads[stage_idx])
        else:
            attention = BiLevelRoutingAttention(
                self.embed_dims[stage_idx],
                self.num_heads[stage_idx],
                window_size=self.window_size,
                topk=self.topks[stage_idx])
        return CompareAttentionBlock(
            dim=self.embed_dims[stage_idx],
            attention=attention,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path)


__all__ = ["CompareBiFormer"]
