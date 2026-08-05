"""对比实验专用的注意力计时工具。

本目录中的模型均使用这里的计时器记录 S1-S4 四个阶段的注意力耗时。
默认关闭计时，训练和普通推理不会引入额外同步开销；开启后每隔若干张图
输出一次按单图归一化的阶段平均耗时。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE_NAMES = ("S1", "S2", "S3", "S4")


class CompareAttentionTimer:
    """累计四个阶段的注意力耗时。

    计时使用事件包住每个注意力模块，但只在一次主干前向结束时同步，避免
    在每个模块之后频繁同步。报告中的单位是单张图毫秒。
    """

    def __init__(self,
                 model_name: str,
                 enabled: bool = False,
                 interval: int = 100) -> None:
        self.model_name = model_name
        self.enabled = bool(enabled)
        self.interval = max(int(interval), 1)
        self._pending_cuda = OrderedDict()
        self._pending_cpu = OrderedDict()
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        self._last_report = None
        self._reports = []
        self._last_batch_size = 1

    def configure(self,
                  enabled: Optional[bool] = None,
                  interval: Optional[int] = None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if interval is not None:
            self.interval = max(int(interval), 1)

    def reset(self) -> None:
        self._pending_cuda.clear()
        self._pending_cpu.clear()
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        self._last_report = None
        self._reports.clear()
        self._last_batch_size = 1

    def begin_forward(self) -> None:
        if not self.enabled:
            return
        self._pending_cuda.clear()
        self._pending_cpu.clear()

    def measure(self,
                stage: str,
                fn: Callable[[], torch.Tensor],
                batch_size: int = 1,
                tensor: Optional[torch.Tensor] = None):
        """执行注意力函数并记录其耗时。"""
        if not self.enabled or stage not in STAGE_NAMES:
            return fn()

        batch_size = max(int(batch_size), 1)
        is_capturing = (
            torch.cuda.is_current_stream_capturing()
            if torch.cuda.is_available() else False)
        use_cuda_event = (
            tensor is not None and tensor.is_cuda and torch.cuda.is_available()
            and not is_capturing)
        if use_cuda_event:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(tensor.device))
            output = fn()
            end.record(torch.cuda.current_stream(tensor.device))
            self._pending_cuda.setdefault(stage, []).append(
                (start, end, batch_size))
            return output

        start_time = time.perf_counter()
        output = fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._pending_cpu.setdefault(stage, []).append(
            (elapsed_ms, batch_size))
        return output

    def _consume_pending(self) -> None:
        """结算当前一次主干前向中各阶段的总注意力耗时。"""
        per_forward = {stage: 0.0 for stage in STAGE_NAMES}
        has_sample = {stage: False for stage in STAGE_NAMES}
        for stage, events in self._pending_cuda.items():
            for start, end, event_batch_size in events:
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                per_forward[stage] += elapsed_ms / max(event_batch_size, 1)
                has_sample[stage] = True
        for stage, records in self._pending_cpu.items():
            for elapsed_ms, event_batch_size in records:
                per_forward[stage] += elapsed_ms / max(event_batch_size, 1)
                has_sample[stage] = True
        for stage in STAGE_NAMES:
            if has_sample[stage]:
                self._sums[stage] += per_forward[stage]
                self._samples[stage] += 1
        self._pending_cuda.clear()
        self._pending_cpu.clear()

    def finish_forward(self, batch_size: int = 1) -> Optional[Dict[str, float]]:
        if not self.enabled:
            return None

        # 这里累加的是一个阶段内所有 Transformer-Block 注意力的总时间，
        # 最终报告为单张图片的阶段平均耗时。
        self._consume_pending()
        self._last_batch_size = max(int(batch_size), 1)
        self._image_count += self._last_batch_size
        if self._image_count < self.interval:
            return None
        return self._emit_report()

    def flush(self) -> Optional[Dict[str, float]]:
        """输出尚未达到 interval 的尾部样本。"""
        if not self.enabled:
            return None
        if self._pending_cuda or self._pending_cpu:
            self._consume_pending()
        if self._image_count <= 0:
            return self._last_report
        return self._emit_report()

    def _emit_report(self) -> Dict[str, float]:
        report = OrderedDict()
        for stage in STAGE_NAMES:
            samples = self._samples[stage]
            report[stage] = (self._sums[stage] / samples) if samples else 0.0
        report["images"] = self._image_count
        report["model"] = self.model_name
        self._last_report = dict(report)
        self._reports.append(dict(report))
        text = " ".join(
            f"{stage}_attention={report[stage]:.4f}ms"
            for stage in STAGE_NAMES)
        print(
            f"[COMPARE-ATTN] model={self.model_name} "
            f"images={self._image_count} {text}")
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        return dict(report)

    @property
    def last_report(self) -> Optional[Dict[str, float]]:
        return self._last_report

    @property
    def reports(self):
        return list(self._reports)


class DropPath(nn.Module):
    """独立实现，避免对外部版本的依赖。"""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """在通道维上进行层归一化。"""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CompareMlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CompareAttentionBlock(nn.Module):
    """统一的预归一化注意力块。输入输出均为 NCHW。"""

    def __init__(self,
                 dim: int,
                 attention: nn.Module,
                 mlp_ratio: float = 4.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = attention
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = CompareMlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self,
                x: torch.Tensor,
                timer: Optional[CompareAttentionTimer],
                stage: str) -> torch.Tensor:
        x_nhwc = x.permute(0, 2, 3, 1)
        normed = self.norm1(x_nhwc)
        if timer is None:
            attn_out = self.attention(normed)
        else:
            attn_out = timer.measure(
                stage,
                lambda: self.attention(normed),
                batch_size=x.shape[0],
                tensor=x)
        x_nhwc = x_nhwc + self.drop_path(attn_out)
        x_nhwc = x_nhwc + self.drop_path(self.mlp(self.norm2(x_nhwc)))
        return x_nhwc.permute(0, 3, 1, 2).contiguous()


class CompareBackboneBase(nn.Module):
    """四阶段分层主干基类，输出 1/4、1/8、1/16、1/32 特征。"""

    def __init__(self,
                 in_chans: int,
                 embed_dims: Iterable[int],
                 depths: Iterable[int],
                 num_heads: Iterable[int],
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.0,
                 model_name: str = "compare",
                 timing: Optional[dict] = None,
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        if not (len(self.embed_dims) == len(self.depths) ==
                len(self.num_heads) == 4):
            raise ValueError("对比主干必须提供四个阶段的配置")

        timing = timing or {}
        self.compare_model_name = model_name
        self.compare_timer = CompareAttentionTimer(
            model_name=model_name,
            enabled=timing.get("enabled", False),
            interval=timing.get("interval", 100))
        self.init_cfg = init_cfg

        self.downsamples = nn.ModuleList()
        for stage_idx, dim in enumerate(self.embed_dims):
            in_dim = in_chans if stage_idx == 0 else self.embed_dims[stage_idx - 1]
            stride = 4 if stage_idx == 0 else 2
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    LayerNorm2d(dim),
                    nn.GELU()))

        total_blocks = sum(self.depths)
        if total_blocks > 1:
            drop_rates = torch.linspace(0, drop_path_rate,
                                        total_blocks).tolist()
        else:
            drop_rates = [0.0]
        self.stages = nn.ModuleList()
        cursor = 0
        for stage_idx, depth in enumerate(self.depths):
            blocks = []
            for block_idx in range(depth):
                blocks.append(
                    self._make_block(
                        stage_idx=stage_idx,
                        block_idx=block_idx,
                        drop_path=drop_rates[cursor + block_idx],
                        mlp_ratio=mlp_ratio))
            self.stages.append(nn.ModuleList(blocks))
            cursor += depth

        self._init_weights()

    def _make_block(self, stage_idx: int, block_idx: int,
                    drop_path: float, mlp_ratio: float) -> nn.Module:
        raise NotImplementedError

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def set_attention_debug(self,
                            enabled: bool = True,
                            interval: Optional[int] = None) -> None:
        self.compare_timer.configure(enabled=enabled, interval=interval)

    def get_attention_report(self) -> Optional[Dict[str, float]]:
        return self.compare_timer.last_report

    def forward(self, x: torch.Tensor):
        self.compare_timer.begin_forward()
        outputs = []
        for stage_idx, (downsample, blocks) in enumerate(
                zip(self.downsamples, self.stages)):
            x = downsample(x)
            stage_name = STAGE_NAMES[stage_idx]
            for block in blocks:
                x = block(x, self.compare_timer, stage_name)
            outputs.append(x)
        self.compare_timer.finish_forward(x.shape[0])
        return outputs

    def flush_attention_report(self):
        return self.compare_timer.flush()


def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """把 NHWC 特征划分成窗口，并返回原始补齐量。"""
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = x.shape[1:3]
    windows = x.view(batch, padded_h // window_size, window_size,
                     padded_w // window_size, window_size,
                     channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, channels)
    return windows, (pad_h, pad_w)


def window_reverse(windows: torch.Tensor,
                   window_size: int,
                   batch: int,
                   height: int,
                   width: int,
                   pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    padded_h, padded_w = height + pad_h, width + pad_w
    channels = windows.shape[-1]
    x = windows.view(batch, padded_h // window_size,
                     padded_w // window_size, window_size, window_size,
                     channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch, padded_h, padded_w, channels)
    return x[:, :height, :width, :]
