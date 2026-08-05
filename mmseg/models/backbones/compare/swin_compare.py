"""对比实验中的分层窗口注意力主干。"""

from typing import Tuple

import torch
import torch.nn as nn

from mmseg.registry import MODELS

from .attention_timing import (CompareAttentionBlock, CompareBackboneBase,
                                window_partition, window_reverse)


class WindowSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("注意力通道数必须能被头数整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        table_size = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads))
        relative_position_index = self._make_relative_position_index(
            self.window_size)
        self.register_buffer("relative_position_index",
                             relative_position_index,
                             persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _make_relative_position_index(window_size: int) -> torch.Tensor:
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w,
                                             indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        return relative.sum(-1)

    def _attention_mask(self, height: int, width: int,
                        device: torch.device) -> torch.Tensor:
        shift_size = self.shift_size
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        padded_h, padded_w = height + pad_h, width + pad_w
        mask = torch.zeros((1, padded_h, padded_w, 1), device=device)
        h_slices = ((0, -self.window_size),
                    (-self.window_size, -shift_size),
                    (-shift_size, None))
        w_slices = h_slices
        counter = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice[0]:h_slice[1], w_slice[0]:w_slice[1], :] = counter
                counter += 1
        mask_windows, _ = window_partition(mask, self.window_size)
        tokens = self.window_size * self.window_size
        mask_windows = mask_windows.view(-1, tokens)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(
            attention_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        shift_size = self.shift_size
        if min(height, width) <= self.window_size:
            shift_size = 0
        if shift_size:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        windows, pad_hw = window_partition(x, self.window_size)
        tokens = self.window_size * self.window_size
        window_batch = windows.shape[0]
        qkv = self.qkv(windows).reshape(window_batch, tokens, 3,
                                        self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if shift_size:
            mask = self._attention_mask(height, width, x.device)
            windows_per_image = mask.shape[0]
            attention = attention.view(batch, windows_per_image,
                                      self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(window_batch, tokens, channels)
        output = self.proj(output)
        output = window_reverse(output, self.window_size, batch, height, width,
                                pad_hw)
        if shift_size:
            output = torch.roll(output, shifts=(shift_size, shift_size), dims=(1, 2))
        return output


_SWIN_VARIANTS = {
    "tiny": dict(
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24)),
    "small": dict(
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24)),
    "base": dict(
        embed_dims=(128, 256, 512, 1024),
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32)),
}


@MODELS.register_module()
class CompareSwin(CompareBackboneBase):
    """Swin-T、Swin-S、Swin-B 的统一实现。"""

    def __init__(self,
                 variant: str = "tiny",
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 **kwargs) -> None:
        variant = str(variant).lower()
        aliases = {"t": "tiny", "s": "small", "b": "base"}
        variant = aliases.get(variant, variant)
        if variant not in _SWIN_VARIANTS:
            raise KeyError(
                f"未知 Swin 版本 {variant}，可选值为 {tuple(_SWIN_VARIANTS)}")
        spec = dict(_SWIN_VARIANTS[variant])
        embed_dims = kwargs.pop("embed_dims", spec["embed_dims"])
        depths = kwargs.pop("depths", spec["depths"])
        num_heads = kwargs.pop("num_heads", spec["num_heads"])
        self.variant = variant
        self.window_size = int(window_size)
        model_name = kwargs.pop("model_name", f"swin_{variant}")
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
        shift_size = 0 if block_idx % 2 == 0 else self.window_size // 2
        attention = WindowSelfAttention(
            dim=self.embed_dims[stage_idx],
            num_heads=self.num_heads[stage_idx],
            window_size=self.window_size,
            shift_size=shift_size)
        return CompareAttentionBlock(
            dim=self.embed_dims[stage_idx],
            attention=attention,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path)


__all__ = ["CompareSwin"]
