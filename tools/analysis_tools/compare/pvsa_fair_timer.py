"""把原始 PVSA 主干接入统一对比测速口径。"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional

import torch


STAGE_NAMES = ("S1", "S2", "S3", "S4")


class PVSAFairStageTimer:
    """通过前向钩子统计 PVSA 各阶段 PA 模块的总耗时。

    该计时器只用于统一公平基准，不改变原始 PVSA 的内部调试逻辑；原始
    ``tools/analysis_tools/benchmark.py`` 仍然保持原样。
    """

    def __init__(self, backbone, model_name: str = "pvsa",
                 interval: int = 100) -> None:
        self.model_name = model_name
        self.enabled = False
        self.interval = max(int(interval), 1)
        self._pending_cuda = OrderedDict()
        self._pending_cpu = OrderedDict()
        self._starts = {}
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        self._last_report = None
        self._reports = []
        self._hooks = []

        stages = getattr(backbone, "stages", None)
        if stages is None or len(stages) != 4:
            raise AttributeError("PVSA 主干缺少四个 stages，无法建立公平计时钩子")
        for stage_index, stage in enumerate(stages):
            stage_name = STAGE_NAMES[stage_index]
            for block in stage:
                attention = getattr(block, "PA", None)
                if attention is None:
                    continue
                self._hooks.append(
                    attention.register_forward_pre_hook(
                        self._make_pre_hook(stage_name)))
                self._hooks.append(
                    attention.register_forward_hook(
                        self._make_post_hook(stage_name)))

        self._hooks.append(
            backbone.register_forward_pre_hook(self._backbone_pre_hook))
        self._hooks.append(
            backbone.register_forward_hook(self._backbone_post_hook))
        self.backbone = backbone

    @staticmethod
    def _is_capturing() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            return False

    def _make_pre_hook(self, stage: str):
        def hook(module, inputs):
            if not self.enabled or not inputs:
                return
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                return
            if tensor.is_cuda and not self._is_capturing():
                start = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(tensor.device))
                self._starts[id(module)] = (stage, start, tensor.shape[0])
            elif not tensor.is_cuda:
                self._starts[id(module)] = (stage, time.perf_counter(),
                                             tensor.shape[0])
        return hook

    def _make_post_hook(self, stage: str):
        def hook(module, inputs, output):
            record = self._starts.pop(id(module), None)
            if record is None or not self.enabled:
                return
            record_stage, start, batch_size = record
            if record_stage != stage:
                return
            batch_size = max(int(batch_size), 1)
            if isinstance(start, torch.cuda.Event):
                if self._is_capturing():
                    return
                end = torch.cuda.Event(enable_timing=True)
                end.record(torch.cuda.current_stream())
                self._pending_cuda.setdefault(stage, []).append(
                    (start, end, batch_size))
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._pending_cpu.setdefault(stage, []).append(
                    (elapsed_ms, batch_size))
        return hook

    def _backbone_pre_hook(self, module, inputs):
        if self.enabled:
            self._pending_cuda.clear()
            self._pending_cpu.clear()
            self._starts.clear()

    def _backbone_post_hook(self, module, inputs, output):
        if not self.enabled:
            return
        batch_size = 1
        if inputs and isinstance(inputs[0], torch.Tensor):
            batch_size = inputs[0].shape[0]
        self._consume_pending()
        self._image_count += max(int(batch_size), 1)
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
        self._starts.clear()
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        self._last_report = None
        self._reports.clear()

    def _consume_pending(self) -> None:
        per_forward = {stage: 0.0 for stage in STAGE_NAMES}
        has_sample = {stage: False for stage in STAGE_NAMES}
        for stage, events in self._pending_cuda.items():
            for start, end, batch_size in events:
                end.synchronize()
                per_forward[stage] += float(start.elapsed_time(end)) / batch_size
                has_sample[stage] = True
        for stage, records in self._pending_cpu.items():
            for elapsed_ms, batch_size in records:
                per_forward[stage] += elapsed_ms / batch_size
                has_sample[stage] = True
        for stage in STAGE_NAMES:
            if has_sample[stage]:
                self._sums[stage] += per_forward[stage]
                self._samples[stage] += 1
        self._pending_cuda.clear()
        self._pending_cpu.clear()

    def _emit_report(self) -> Dict[str, float]:
        report = OrderedDict()
        for stage in STAGE_NAMES:
            count = self._samples[stage]
            report[stage] = self._sums[stage] / count if count else 0.0
        report["images"] = self._image_count
        report["model"] = self.model_name
        self._last_report = dict(report)
        self._reports.append(dict(report))
        text = " ".join(
            f"{stage}_attention={report[stage]:.4f}ms"
            for stage in STAGE_NAMES)
        print(f"[COMPARE-ATTN] model={self.model_name} "
              f"images={self._image_count} {text}")
        self._sums = {stage: 0.0 for stage in STAGE_NAMES}
        self._samples = {stage: 0 for stage in STAGE_NAMES}
        self._image_count = 0
        return dict(report)

    def flush(self):
        if not self.enabled:
            return self._last_report
        if self._pending_cuda or self._pending_cpu:
            self._consume_pending()
        if self._image_count > 0:
            return self._emit_report()
        return self._last_report

    @property
    def reports(self):
        return list(self._reports)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
