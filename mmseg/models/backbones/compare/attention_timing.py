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

from tools.analysis_tools.compare.cuda_graph_timing import new_graph_event


STAGE_NAMES = ("S1", "S2", "S3", "S4")


class CompareAttentionTimer:
    """累计四个阶段的注意力耗时。

    普通前向使用 CUDA Event 包住注意力模块和完整 stage；CUDA Graph 模式下，
    事件会在捕获阶段作为图内节点记录，重放完成后在图外读取 elapsed_time。
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
        self._pending_total_cuda = OrderedDict()
        self._pending_total_cpu = OrderedDict()
        self._graph_attention_cuda = OrderedDict()
        self._graph_total_cuda = OrderedDict()
        self._graph_capture = False
        self._timing_mode = "eager"
        self._graph_event_backend = None
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._total_sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._total_samples = {stage: 0 for stage in STAGE_NAMES}
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
        self._pending_total_cuda.clear()
        self._pending_total_cpu.clear()
        self._graph_attention_cuda.clear()
        self._graph_total_cuda.clear()
        self._graph_capture = False
        self._timing_mode = "eager"
        self._graph_event_backend = None
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._total_sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._total_samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        self._last_report = None
        self._reports.clear()
        self._last_batch_size = 1

    @staticmethod
    def _is_capturing() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            return False

    def _new_event(self, graph_capture: bool = False):
        if not graph_capture:
            return torch.cuda.Event(enable_timing=True)
        event = new_graph_event()
        self._graph_event_backend = getattr(
            event, "backend", "pytorch_external")
        return event

    def validate_graph_timing_support(self) -> None:
        """在正式捕获前验证图内事件接口。"""
        self._new_event(graph_capture=True)

    def begin_graph_capture(self) -> None:
        if not self.enabled:
            raise RuntimeError("启用 CUDA Graph 阶段计时前必须先启用计时器")
        self._pending_cuda.clear()
        self._pending_cpu.clear()
        self._pending_total_cuda.clear()
        self._pending_total_cpu.clear()
        self._graph_attention_cuda.clear()
        self._graph_total_cuda.clear()
        # 在正式捕获前先验证当前 PyTorch 是否支持 external event。
        self._new_event(graph_capture=True)
        self._timing_mode = "cuda_graph"
        self._graph_capture = True

    def end_graph_capture(self) -> None:
        if not self._graph_capture:
            return
        self._graph_capture = False
        if not self._graph_attention_cuda and not self._graph_total_cuda:
            raise RuntimeError(
                "CUDA Graph 捕获完成，但没有记录到阶段事件；请确认模型使用了 "
                "统一对比计时器。")

    def begin_forward(self) -> None:
        """开始一次普通前向，清理上一轮尚未结算的事件。"""
        if not self.enabled:
            return
        self._pending_cuda.clear()
        self._pending_cpu.clear()
        self._pending_total_cuda.clear()
        self._pending_total_cpu.clear()

    def begin_stage(self, stage: str, tensor: Optional[torch.Tensor] = None):
        """开始记录一个完整 stage（下采样 + Transformer Block）。"""
        if not self.enabled or stage not in STAGE_NAMES:
            return None
        if self._graph_capture:
            if tensor is None or not tensor.is_cuda:
                raise RuntimeError("CUDA Graph 阶段计时需要 CUDA 输入张量")
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event(graph_capture=True)
            start.record(stream)
            return ("graph", start, stream, max(int(tensor.shape[0]), 1))
        is_capturing = self._is_capturing()
        if tensor is not None and tensor.is_cuda and not is_capturing:
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event()
            start.record(stream)
            return ("cuda", start, stream, max(int(tensor.shape[0]), 1))
        return ("cpu", time.perf_counter(),
                max(int(tensor.shape[0]), 1) if tensor is not None else 1)

    def end_stage(self, stage: str, token, batch_size: int = 1) -> None:
        if token is None or not self.enabled or stage not in STAGE_NAMES:
            return
        kind = token[0]
        batch_size = max(int(batch_size), 1)
        if kind in ("cuda", "graph"):
            _, start, stream, _ = token
            end = self._new_event(graph_capture=(kind == "graph"))
            end.record(stream)
            target = (self._graph_total_cuda
                      if kind == "graph" else self._pending_total_cuda)
            target.setdefault(stage, []).append((start, end, batch_size))
        else:
            _, start_time, _ = token
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._pending_total_cpu.setdefault(stage, []).append(
                (elapsed_ms, batch_size))

    def measure(self,
                stage: str,
                fn: Callable[[], torch.Tensor],
                batch_size: int = 1,
                tensor: Optional[torch.Tensor] = None):
        """执行注意力函数并记录其耗时。"""
        if not self.enabled or stage not in STAGE_NAMES:
            return fn()

        batch_size = max(int(batch_size), 1)
        if self._graph_capture:
            if tensor is None or not tensor.is_cuda:
                raise RuntimeError("CUDA Graph 注意力计时需要 CUDA 输入张量")
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event(graph_capture=True)
            start.record(stream)
            output = fn()
            end = self._new_event(graph_capture=True)
            end.record(stream)
            self._graph_attention_cuda.setdefault(stage, []).append(
                (start, end, batch_size))
            return output

        is_capturing = self._is_capturing()
        use_cuda_event = (
            tensor is not None and tensor.is_cuda and torch.cuda.is_available()
            and not is_capturing)
        if use_cuda_event:
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event()
            end = self._new_event()
            start.record(stream)
            output = fn()
            end.record(stream)
            self._pending_cuda.setdefault(stage, []).append(
                (start, end, batch_size))
            return output

        start_time = time.perf_counter()
        output = fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._pending_cpu.setdefault(stage, []).append(
            (elapsed_ms, batch_size))
        return output

    @staticmethod
    def _collect_pending(pending_cuda, pending_cpu):
        per_forward = {stage: 0.0 for stage in STAGE_NAMES}
        has_sample = {stage: False for stage in STAGE_NAMES}
        for stage, events in pending_cuda.items():
            for start, end, event_batch_size in events:
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                per_forward[stage] += elapsed_ms / max(event_batch_size, 1)
                has_sample[stage] = True
        for stage, records in pending_cpu.items():
            for elapsed_ms, event_batch_size in records:
                per_forward[stage] += elapsed_ms / max(event_batch_size, 1)
                has_sample[stage] = True
        return per_forward, has_sample

    def _accumulate(self, attention, has_attention, stage_total,
                    has_stage_total, batch_size: int) -> None:
        for stage in STAGE_NAMES:
            if has_attention[stage]:
                self._sums[stage] += attention[stage]
                self._samples[stage] += 1
            if has_stage_total[stage]:
                self._total_sums[stage] += stage_total[stage]
                self._total_samples[stage] += 1
        self._image_count += max(int(batch_size), 1)

    def _consume_pending(self) -> None:
        """结算普通前向的注意力耗时和完整 stage 耗时。"""
        attention, has_attention = self._collect_pending(
            self._pending_cuda, self._pending_cpu)
        stage_total, has_stage_total = self._collect_pending(
            self._pending_total_cuda, self._pending_total_cpu)
        self._accumulate(attention, has_attention, stage_total,
                         has_stage_total, self._last_batch_size)
        self._pending_cuda.clear()
        self._pending_cpu.clear()
        self._pending_total_cuda.clear()
        self._pending_total_cpu.clear()

    def consume_graph_replay(self, batch_size: int = 1):
        """在一次 CUDA Graph 重放完成后读取图内事件。"""
        if not self.enabled:
            return None
        if self._graph_capture:
            raise RuntimeError("CUDA Graph 尚未结束捕获，不能读取图内事件")
        attention, has_attention = self._collect_pending(
            self._graph_attention_cuda, {})
        stage_total, has_stage_total = self._collect_pending(
            self._graph_total_cuda, {})
        self._accumulate(attention, has_attention, stage_total,
                         has_stage_total, batch_size)
        self._graph_attention_cuda.clear()
        self._graph_total_cuda.clear()
        if self._image_count >= self.interval:
            return self._emit_report()
        return None

    def finish_forward(self, batch_size: int = 1) -> Optional[Dict[str, float]]:
        if not self.enabled or self._graph_capture:
            return None
        self._last_batch_size = max(int(batch_size), 1)
        self._consume_pending()
        if self._image_count >= self.interval:
            return self._emit_report()
        return None

    def flush(self) -> Optional[Dict[str, float]]:
        """输出尚未达到 interval 的尾部样本。"""
        if not self.enabled or self._graph_capture:
            return self._last_report
        if (self._pending_cuda or self._pending_cpu
                or self._pending_total_cuda or self._pending_total_cpu):
            self._last_batch_size = 1
            self._consume_pending()
        if self._image_count <= 0:
            return self._last_report
        return self._emit_report()

    def _emit_report(self) -> Dict[str, float]:
        report = OrderedDict()
        for stage in STAGE_NAMES:
            samples = self._samples[stage]
            total_samples = self._total_samples[stage]
            report[stage] = (self._sums[stage] / samples) if samples else 0.0
            report[f"{stage}_total"] = (
                self._total_sums[stage] / total_samples
                if total_samples else 0.0)
        report["images"] = self._image_count
        report["model"] = self.model_name
        self._last_report = dict(report)
        self._reports.append(dict(report))
        text = " ".join(
            f"{stage}_attention={report[stage]:.4f}ms "
            f"{stage}_total={report[f'{stage}_total']:.4f}ms"
            for stage in STAGE_NAMES)
        tag = ("[COMPARE-CUDA-GRAPH-ATTN]"
               if self._timing_mode == "cuda_graph" else "[COMPARE-ATTN]")
        print(
            f"{tag} model={self.model_name} "
            f"images={self._image_count} {text}")
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._total_sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._total_samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        return dict(report)

    @property
    def last_report(self) -> Optional[Dict[str, float]]:
        return self._last_report

    @property
    def graph_event_backend(self) -> Optional[str]:
        return self._graph_event_backend

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
            stage_name = STAGE_NAMES[stage_idx]
            stage_token = self.compare_timer.begin_stage(stage_name, x)
            x = downsample(x)
            for block in blocks:
                x = block(x, self.compare_timer, stage_name)
            self.compare_timer.end_stage(stage_name, stage_token, x.shape[0])
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
