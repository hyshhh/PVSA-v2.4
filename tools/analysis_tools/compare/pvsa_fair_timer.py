"""把原始 PVSA 主干接入统一对比测速口径。"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional

import torch

try:
    from .cuda_graph_timing import new_graph_event
except ImportError:
    from cuda_graph_timing import new_graph_event


STAGE_NAMES = ("S1", "S2", "S3", "S4")


class PVSAFairStageTimer:
    """通过前向钩子统计 PVSA 注意力和阶段外层总耗时。

    ``S1`` 到 ``S4`` 是各阶段所有 ``PA`` 模块的注意力耗时；
    ``S1_total`` 到 ``S4_total`` 是从该阶段双分支下采样开始，
    到 Transformer 阶段和 FAM 结束的外层耗时。使用阶段边界事件而不是
    简单累加子模块事件，因此不会漏掉分支之间的拼接、调度和其他边界操作。
    该统计包含 CNN 分支、Transformer 下采样、Transformer Block 和 FAM，
    不包含四个 stage 之后的跨阶段融合、输出归一化与解码头。
    该计时器只用于统一公平基准，不改变原始 PVSA 主干代码。
    """

    def __init__(self, backbone, model_name: str = "pvsa",
                 interval: int = 100) -> None:
        self.model_name = model_name
        self.enabled = False
        self.interval = max(int(interval), 1)
        self._pending_cuda = OrderedDict()
        self._pending_cpu = OrderedDict()
        self._pending_total_cuda = OrderedDict()
        self._pending_total_cpu = OrderedDict()
        self._starts = {}
        self._total_starts = {}
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
        self._hooks = []
        self.backbone = backbone

        stages = getattr(backbone, "stages", None)
        if stages is None or len(stages) != 4:
            raise AttributeError("PVSA 主干缺少四个 stages，无法建立公平计时钩子")
        for stage_index, stage in enumerate(stages):
            stage_name = STAGE_NAMES[stage_index]

            # 不假定 stage 一定是可迭代的 Sequential。使用递归模块遍历，
            # 兼容 checkpoint_wrapper 等包装模块，同时保证每个 PA 只挂一次钩子。
            seen_attention = set()
            for block in stage.modules():
                attention = getattr(block, "PA", None)
                if (attention is None
                        or not hasattr(attention, "register_forward_pre_hook")
                        or id(attention) in seen_attention):
                    continue
                seen_attention.add(id(attention))
                self._hooks.append(
                    attention.register_forward_pre_hook(
                        self._make_pre_hook(stage_name)))
                self._hooks.append(
                    attention.register_forward_hook(
                        self._make_post_hook(stage_name)))

            # 阶段总耗时采用一个外层边界：从 CNN 分支下采样开始，到 FAM
            # 结束；没有 FAM 时退化为 Transformer stage 结束。这样会把
            # 两条分支之间的拼接、调度等边界操作纳入，而不会把多个子模块
            # 的事件简单相加后误称为阶段墙钟耗时。跨阶段融合在四个 stage
            # 全部完成后执行，因此仍不归入单个 stage。
            start_module = self._get_stage_module(
                getattr(backbone, "downsample_layers2", None), stage_index)
            if start_module is None:
                start_module = self._get_stage_module(
                    getattr(backbone, "downsample_layers", None), stage_index)
            end_module = self._get_stage_module(
                getattr(backbone, "FAM", None), stage_index)
            if end_module is None:
                end_module = stage
            if start_module is not None and end_module is not None:
                self._hooks.append(
                    start_module.register_forward_pre_hook(
                        self._make_total_pre_hook(stage_name)))
                self._hooks.append(
                    end_module.register_forward_hook(
                        self._make_total_post_hook(stage_name)))

        self._hooks.append(
            backbone.register_forward_pre_hook(self._backbone_pre_hook))
        self._hooks.append(
            backbone.register_forward_hook(self._backbone_post_hook))

    @staticmethod
    def _get_stage_module(container, index: int):
        if container is None:
            return None
        try:
            module = container[index]
        except (IndexError, KeyError, TypeError):
            return None
        if not hasattr(module, "register_forward_pre_hook"):
            return None
        return module

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
        self._new_event(graph_capture=True)
        self._timing_mode = "cuda_graph"
        self._graph_capture = True

    def end_graph_capture(self) -> None:
        if not self._graph_capture:
            return
        self._graph_capture = False
        if not self._graph_attention_cuda and not self._graph_total_cuda:
            raise RuntimeError(
                "CUDA Graph 捕获完成，但没有记录到 PVSA 阶段事件；请确认模型 "
                "使用了 PVSA 公平计时器。")

    def _start_record(self, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return None
        tensor = inputs[0]
        if tensor.is_cuda and self._graph_capture:
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event(graph_capture=True)
            start.record(stream)
            return ("graph", start, stream, max(int(tensor.shape[0]), 1))
        if tensor.is_cuda and not self._is_capturing():
            stream = torch.cuda.current_stream(tensor.device)
            start = self._new_event()
            start.record(stream)
            return ("cuda", start, stream, max(int(tensor.shape[0]), 1))
        if not tensor.is_cuda:
            return ("cpu", time.perf_counter(),
                    max(int(tensor.shape[0]), 1))
        return None

    def _finish_record(self, record, stage, pending_cuda, pending_cpu,
                       graph_cuda=None):
        if record is None:
            return
        kind = record[0]
        if kind in ("cuda", "graph"):
            _, start, stream, batch_size = record
            end = self._new_event(graph_capture=(kind == "graph"))
            end.record(stream)
            target = (graph_cuda
                      if kind == "graph" else pending_cuda)
            if target is not None:
                target.setdefault(stage, []).append(
                    (start, end, max(int(batch_size), 1)))
        else:
            _, start_time, batch_size = record
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            pending_cpu.setdefault(stage, []).append(
                (elapsed_ms, max(int(batch_size), 1)))

    def _make_pre_hook(self, stage: str):
        def hook(module, inputs):
            if self.enabled:
                self._starts[id(module)] = self._start_record(inputs)
        return hook

    def _make_post_hook(self, stage: str):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            record = self._starts.pop(id(module), None)
            self._finish_record(
                record, stage, self._pending_cuda, self._pending_cpu,
                self._graph_attention_cuda)
        return hook

    def _make_total_pre_hook(self, stage: str):
        def hook(module, inputs):
            if self.enabled:
                self._total_starts[stage] = self._start_record(inputs)
        return hook

    def _make_total_post_hook(self, stage: str):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            record = self._total_starts.pop(stage, None)
            self._finish_record(
                record, stage, self._pending_total_cuda,
                self._pending_total_cpu, self._graph_total_cuda)
        return hook

    def _backbone_pre_hook(self, module, inputs):
        if self.enabled:
            self._pending_cuda.clear()
            self._pending_cpu.clear()
            self._pending_total_cuda.clear()
            self._pending_total_cpu.clear()
            self._starts.clear()
            self._total_starts.clear()

    def _backbone_post_hook(self, module, inputs, output):
        if not self.enabled or self._graph_capture:
            return
        batch_size = 1
        if inputs and isinstance(inputs[0], torch.Tensor):
            batch_size = inputs[0].shape[0]
        self._last_batch_size = max(int(batch_size), 1)
        self._consume_pending()
        if self._image_count >= self.interval:
            self._emit_report()

    def configure(self, enabled: Optional[bool] = None,
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
        self._starts.clear()
        self._total_starts.clear()
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
    def _collect_pending(pending_cuda, pending_cpu):
        per_forward = {stage: 0.0 for stage in STAGE_NAMES}
        has_sample = {stage: False for stage in STAGE_NAMES}
        for stage, events in pending_cuda.items():
            for start, end, batch_size in events:
                end.synchronize()
                per_forward[stage] += float(start.elapsed_time(end)) / batch_size
                has_sample[stage] = True
        for stage, records in pending_cpu.items():
            for elapsed_ms, batch_size in records:
                per_forward[stage] += elapsed_ms / batch_size
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
        attention, has_attention = self._collect_pending(
            self._pending_cuda, self._pending_cpu)
        stage_total, has_stage_total = self._collect_pending(
            self._pending_total_cuda, self._pending_total_cpu)
        batch_size = self._last_batch_size if hasattr(self, '_last_batch_size') else 1
        self._accumulate(attention, has_attention, stage_total,
                         has_stage_total, batch_size)
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

    def _emit_report(self) -> Dict[str, float]:
        report = OrderedDict()
        for stage in STAGE_NAMES:
            count = self._samples[stage]
            total_count = self._total_samples[stage]
            report[stage] = self._sums[stage] / count if count else 0.0
            report[f"{stage}_total"] = (
                self._total_sums[stage] / total_count
                if total_count else 0.0)
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
        print(f"{tag} model={self.model_name} "
              f"images={self._image_count} {text}")
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._total_sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._total_samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        return dict(report)

    def flush(self):
        if not self.enabled:
            return self._last_report
        if (self._pending_cuda or self._pending_cpu
                or self._pending_total_cuda or self._pending_total_cpu):
            self._consume_pending()
        if self._image_count > 0:
            return self._emit_report()
        return self._last_report

    @property
    def graph_event_backend(self) -> Optional[str]:
        return self._graph_event_backend

    @property
    def reports(self):
        return list(self._reports)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
