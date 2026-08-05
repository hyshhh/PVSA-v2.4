"""对比实验测速脚本的公共入口。

该脚本只构建对比实验配置中的模型，并以固定输入统计整体吞吐和 S1-S4
注意力耗时；不依赖数据集，便于在相同显卡和输入尺寸下公平比较。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import mmseg.models.backbones.compare  # noqa: F401



def parse_args():
    parser = argparse.ArgumentParser(description="对比模型 FPS 与阶段注意力测速")
    parser.add_argument("config", help="对比实验配置文件")
    parser.add_argument("checkpoint", nargs="?", default=None, help="可选权重")
    parser.add_argument("--input-size", type=int, nargs=2, default=(224, 224),
                        metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="输出 S1-S4 阶段耗时")
    parser.add_argument("--debug-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="保存结果的 JSON 路径")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser.parse_args()


def _build_model(cfg: Config, checkpoint: Optional[str], device: torch.device):
    model_cfg = cfg.model
    model_cfg.setdefault("train_cfg", None)
    model = MODELS.build(model_cfg)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def _set_timer(model, enabled: bool, interval: int):
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "set_attention_debug"):
        backbone.set_attention_debug(enabled=enabled, interval=interval)
        backbone.compare_timer.reset()
    elif hasattr(backbone, "compare_timer"):
        backbone.compare_timer.configure(enabled=enabled, interval=interval)
        backbone.compare_timer.reset()
    else:
        raise AttributeError("配置中的主干不是对比实验主干，缺少阶段计时器")


def _flush_timer(model):
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "flush_attention_report"):
        return backbone.flush_attention_report()
    return None


def _forward_model(model, inputs: torch.Tensor):
    """统计完整分割网络前向（主干 + 解码头），不包含后处理。"""
    with torch.inference_mode():
        return model._forward(inputs, None)


def _measure_once(model, inputs: torch.Tensor, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    _forward_model(model, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _run_one(model, inputs, device, args, run_index: int) -> Dict:
    # 预热阶段不计入阶段耗时，避免把首次算子选择和缓存建立计入结果。
    _set_timer(model, False, args.debug_interval)
    for _ in range(max(args.warmup, 0)):
        _forward_model(model, inputs)
    _set_timer(model, args.debug, args.debug_interval)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = []
    for index in range(max(args.iters, 1)):
        elapsed.append(_measure_once(model, inputs, device))
        if (index + 1) % max(args.debug_interval, 1) == 0:
            fps = args.batch_size * (index + 1) / sum(elapsed)
            print(f"[COMPARE-FPS] run={run_index} iter={index + 1} "
                  f"fps={fps:.4f} img/s")
    _flush_timer(model)
    backbone = getattr(model, "backbone", model)
    attention_reports = (
        backbone.compare_timer.reports
        if hasattr(backbone, "compare_timer") else [])

    total_seconds = sum(elapsed)
    fps = args.batch_size * len(elapsed) / total_seconds
    latency_ms = total_seconds / len(elapsed) * 1000.0
    result = dict(
        run=run_index,
        fps=fps,
        latency_ms_per_batch=latency_ms,
        latency_ms_per_image=latency_ms / max(args.batch_size, 1),
        images=args.batch_size * len(elapsed),
        attention_reports=attention_reports)
    print(f"[COMPARE-FPS] run={run_index} overall_fps={fps:.4f} img/s "
          f"latency={latency_ms:.4f}ms/batch")
    return result


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.iters <= 0:
        raise ValueError("--batch-size 和 --iters 必须为正数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("未检测到显卡，自动切换到 CPU；CPU 结果仅用于功能检查")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    model = _build_model(cfg, args.checkpoint, device)
    height, width = args.input_size
    inputs = torch.randn(args.batch_size, 3, height, width, device=device)
    print(f"[COMPARE] model={cfg.model.backbone.get('model_name', 'compare')} "
          f"input=({height}, {width}) batch={args.batch_size} device={device}")

    runs = [
        _run_one(model, inputs, device, args, run_index)
        for run_index in range(1, args.repeat_times + 1)]
    average_fps = sum(item["fps"] for item in runs) / len(runs)
    result = dict(
        config=os.path.abspath(args.config),
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else None,
        input_size=[height, width],
        batch_size=args.batch_size,
        debug=args.debug,
        runs=runs,
        average_fps=average_fps)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[COMPARE] 结果已保存：{output_path}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
