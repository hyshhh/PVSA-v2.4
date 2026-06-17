# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradSpikeDebugHook(Hook):
    """Log parameters with the largest gradients when a spike appears."""

    def __init__(self, threshold=1e5, topk=10, interval=1,
                 sanitize_nonfinite=True):
        self.threshold = float(threshold)
        self.topk = int(topk)
        self.interval = int(interval)
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self._handles = []
        self._grad_items = []
        self._nonfinite_items = []

    def before_train(self, runner):
        if self._handles:
            return

        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue

            def save_grad(grad, name=name, shape=tuple(param.shape)):
                finite_mask = torch.isfinite(grad)
                has_nonfinite = not bool(finite_mask.all().item())
                safe_grad = grad
                if has_nonfinite:
                    nonfinite_count = int((~finite_mask).sum().item())
                    self._nonfinite_items.append(
                        (nonfinite_count, name, shape))
                    if self.sanitize_nonfinite:
                        safe_grad = torch.nan_to_num(
                            grad, nan=0.0, posinf=0.0, neginf=0.0)

                norm = torch.linalg.vector_norm(
                    safe_grad.detach().float()).item()
                if not math.isfinite(norm):
                    norm = float('inf')
                self._grad_items.append((norm, name, shape))
                return safe_grad

            self._handles.append(param.register_hook(save_grad))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._grad_items = []
        self._nonfinite_items = []

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.interval > 1 and (runner.iter + 1) % self.interval != 0:
            return

        total_sq = 0.0
        for norm, _, _ in self._grad_items:
            total_sq += norm * norm

        if not self._grad_items:
            return
        total_norm = math.sqrt(total_sq)
        if math.isfinite(total_norm) and total_norm < self.threshold:
            return

        grad_items = sorted(
            self._grad_items, key=lambda item: item[0], reverse=True)
        summary = ', '.join(
            f'{name}: norm={norm:.4e}, shape={shape}'
            for norm, name, shape in grad_items[:self.topk])
        message = (
            f'[GradSpike] iter={runner.iter + 1} '
            f'total_norm={total_norm:.4e} top{self.topk}: {summary}')
        if self._nonfinite_items:
            nonfinite_items = sorted(
                self._nonfinite_items, key=lambda item: item[0],
                reverse=True)
            nonfinite_summary = ', '.join(
                f'{name}: nonfinite={count}, shape={shape}'
                for count, name, shape in nonfinite_items[:self.topk])
            message = (
                f'{message} | nonfinite_grad_sanitized='
                f'{self.sanitize_nonfinite} top{self.topk}: '
                f'{nonfinite_summary}')
        runner.logger.warning(message)

    def after_train(self, runner):
        for handle in self._handles:
            handle.remove()
        self._handles = []
