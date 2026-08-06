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
from mmengine.config import DictAction
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import mmseg.models.backbones.compare  # noqa: F401

from pvsa_fair_timer import PVSAFairStageTimer


def _str_to_bool(value):
    """把命令行中的 true/false/1/0 转成布尔值。"""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError("布尔参数只能填写 true/false 或 1/0")


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
    parser.add_argument(
        "--debug",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "是否输出 S1-S4 阶段耗时，默认 false；"
            "可写 --debug、--debug true/false 或 --debug 1/0。"))
    parser.add_argument("--debug-interval", type=int, default=100)
    parser.add_argument(
        "--cuda-graph",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "是否使用固定输入捕获并重放 CUDA Graph，默认 true；"
            "可显式写 --cuda-graph true 或 --cuda-graph false。"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="保存结果的 JSON 路径")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="覆盖配置中的键值，例如 model.backbone.topp_flash_backend=cuda")
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


def _is_pvsa_model(cfg: Config) -> bool:
    backbone = cfg.model.get("backbone", {})
    return backbone.get("type") in ("BiFormer_fusion", "VTFormer")


def _validate_cuda_graph_compatibility(cfg: Config, args) -> None:
    """在捕获前拒绝 PVSA 的动态路由路径，避免污染 CUDA 捕获状态。"""
    if not args.cuda_graph or not _is_pvsa_model(cfg):
        return

    backbone_cfg = cfg.model.get("backbone", {})
    backend = backbone_cfg.get("topp_flash_backend", None)
    normalized = (str(backend).strip().lower()
                  if backend is not None else "")
    if normalized not in ("cuda", "cuda_forward"):
        raise RuntimeError(
            "--cuda-graph true 需要 PVSA 使用 CUDA 后端；当前 "
            "model.backbone.topp_flash_backend="
            f"{backend!r}。PVSA 的 torch 路径包含动态路由形状，不能被 CUDA "
            "Graph 捕获。请改用 --cuda-graph false，或设置 "
            "model.backbone.topp_flash_backend=cuda 并确认自定义 CUDA 扩展已编译。")


def _attach_model_timer(model, cfg: Config, interval: int):
    """为原始 PVSA 主干安装统一测速所需的计时接口。"""
    if not _is_pvsa_model(cfg):
        return None
    backbone = model.backbone
    timer = PVSAFairStageTimer(
        backbone,
        model_name=cfg.model.backbone.get("model_name", "pvsa"),
        interval=interval)
    # 将 PVSA 专用钩子计时器适配到统一测速接口。
    backbone.compare_timer = timer
    backbone.set_attention_debug = timer.configure
    backbone.flush_attention_report = timer.flush
    backbone._pvsa_fair_stage_timer = timer
    return timer


def _get_attention_timer(model):
    backbone = getattr(model, "backbone", model)
    timer = getattr(backbone, "compare_timer", None)
    if timer is None:
        raise AttributeError("配置中的主干缺少统一阶段计时器")
    return timer


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


def _attention_reports(model):
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "compare_timer"):
        return backbone.compare_timer.reports
    return []


def _run_eager(model, inputs, device, args, run_index: int,
               report_prefix="") -> Dict:
    """普通前向测速，可选同步统计 S1-S4 阶段耗时。"""
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
            prefix = f"{report_prefix} " if report_prefix else ""
            print(f"[COMPARE-FPS] {prefix}run={run_index} "
                  f"iter={index + 1} fps={fps:.4f} img/s")
    _flush_timer(model)
    attention_reports = _attention_reports(model)

    total_seconds = sum(elapsed)
    fps = args.batch_size * len(elapsed) / total_seconds
    latency_ms = total_seconds / len(elapsed) * 1000.0
    prefix = f"{report_prefix} " if report_prefix else ""
    print(f"[COMPARE-FPS] {prefix}run={run_index} "
          f"overall_fps={fps:.4f} img/s latency={latency_ms:.4f}ms/batch")
    return dict(
        run=run_index,
        fps=fps,
        latency_ms_per_batch=latency_ms,
        latency_ms_per_image=latency_ms / max(args.batch_size, 1),
        images=args.batch_size * len(elapsed),
        attention_reports=attention_reports,
        mode="eager")


class _CudaGraphForward:
    """固定输入尺寸的完整分割网络 CUDA Graph 捕获器。"""

    def __init__(self, model, inputs: torch.Tensor, warmup: int,
                 device: torch.device, capture_timing: bool = False,
                 timer=None, timer_interval: int = 100) -> None:
        if device.type != "cuda":
            raise RuntimeError("CUDA Graph 需要显卡设备")
        if capture_timing and timer is None:
            raise RuntimeError("CUDA Graph 阶段计时缺少统一计时器")
        self.model = model
        self.device = device
        self.graph_input = inputs.detach().clone()
        self.graph = torch.cuda.CUDAGraph()
        self.graph_output = None
        self.capture_timing = bool(capture_timing)
        self.timer = timer

        # 预热不插入图内事件，避免把预热阶段计入报告。
        _set_timer(model, False, 1)
        with torch.inference_mode():
            for _ in range(max(int(warmup), 1)):
                model._forward(self.graph_input, None)
        torch.cuda.synchronize(device)

        if self.capture_timing:
            timer.reset()
            timer.configure(enabled=True, interval=timer_interval)
            timer.begin_graph_capture()

        try:
            with torch.inference_mode():
                with torch.cuda.graph(self.graph):
                    self.graph_output = model._forward(self.graph_input, None)
            torch.cuda.synchronize(device)
        except Exception as exc:
            if self.capture_timing:
                timer._graph_capture = False
            raise RuntimeError(
                "对比模型 CUDA Graph 捕获失败，请确认输入尺寸固定、模型处于 eval 模式，"
                "并检查显卡、框架版本和自定义 CUDA 后端是否支持 CUDA Graph。") from exc
        finally:
            if self.capture_timing:
                timer.end_graph_capture()
        print("[COMPARE-CUDA-GRAPH] 捕获完成" +
              ("（含图内阶段事件）" if self.capture_timing else ""))

    def replay(self, inputs: torch.Tensor) -> None:
        # CUDA Graph 捕获阶段使用了 inference_mode；重放也必须处于同一模式，
        # 否则较新版本的 PyTorch 会把图内静态输出视为 inference tensor。
        with torch.inference_mode():
            if inputs.data_ptr() != self.graph_input.data_ptr():
                self.graph_input.copy_(inputs)
            self.graph.replay()

    def consume_timing(self, batch_size: int):
        if not self.capture_timing:
            return None
        return self.timer.consume_graph_replay(batch_size)

    def flush_timing(self):
        if not self.capture_timing:
            return None
        return self.timer.flush()


def _measure_graph_once(graph_runner: _CudaGraphForward,
                        inputs: torch.Tensor,
                        device: torch.device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    graph_runner.replay(inputs)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _run_graph_loop(graph_runner: _CudaGraphForward, inputs, device, args,
                    run_index: int, label: str) -> Dict:
    elapsed = []
    for index in range(max(args.iters, 1)):
        elapsed.append(_measure_graph_once(graph_runner, inputs, device))
        graph_runner.consume_timing(args.batch_size)
        if (index + 1) % max(args.debug_interval, 1) == 0:
            fps = args.batch_size * (index + 1) / sum(elapsed)
            print(f"[{label}] run={run_index} iter={index + 1} "
                  f"fps={fps:.4f} img/s")
    graph_runner.flush_timing()

    total_seconds = sum(elapsed)
    fps = args.batch_size * len(elapsed) / total_seconds
    latency_ms = total_seconds / len(elapsed) * 1000.0
    print(f"[{label}] run={run_index} overall_fps={fps:.4f} img/s "
          f"latency={latency_ms:.4f}ms/batch")
    return dict(
        run=run_index,
        fps=fps,
        latency_ms_per_batch=latency_ms,
        latency_ms_per_image=latency_ms / max(args.batch_size, 1),
        images=args.batch_size * len(elapsed),
        attention_reports=(
            list(graph_runner.timer.reports)
            if graph_runner.capture_timing else []),
        mode=("cuda_graph_attention" if graph_runner.capture_timing
              else "cuda_graph"))


def _run_cuda_graph(model, inputs, device, args, run_index: int) -> Dict:
    """先测干净 CUDA Graph，再按需单独捕获带阶段事件的 Graph。"""
    timer = None
    if args.debug:
        # 在正式测速前检查图内事件接口，避免先完成一轮 FPS 测试后才报错。
        timer = _get_attention_timer(model)
        timer.validate_graph_timing_support()

    clean_runner = _CudaGraphForward(
        model, inputs, args.warmup, device, capture_timing=False)
    result = _run_graph_loop(
        clean_runner, inputs, device, args, run_index, "COMPARE-CUDA-GRAPH")

    if args.debug:
        profile_runner = _CudaGraphForward(
            model,
            inputs,
            args.warmup,
            device,
            capture_timing=True,
            timer=timer,
            timer_interval=args.debug_interval)
        profile_result = _run_graph_loop(
            profile_runner,
            inputs,
            device,
            args,
            run_index,
            "COMPARE-CUDA-GRAPH-ATTN")
        result["attention_reports"] = profile_result["attention_reports"]
        result["graph_attention_profile"] = profile_result
    return result


def _run_one(model, inputs, device, args, run_index: int) -> Dict:
    if not args.cuda_graph:
        return _run_eager(model, inputs, device, args, run_index)
    return _run_cuda_graph(model, inputs, device, args, run_index)

def main():
    args = parse_args()
    if args.batch_size <= 0 or args.iters <= 0:
        raise ValueError("--batch-size 和 --iters 必须为正数")
    if args.debug_interval <= 0:
        raise ValueError("--debug-interval 必须为正数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("未检测到显卡，自动切换到 CPU；CPU 结果仅用于功能检查")
        args.device = "cpu"
    device = torch.device(args.device)
    if args.cuda_graph and device.type != "cuda":
        raise RuntimeError("--cuda-graph 需要可用的 CUDA 显卡")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _validate_cuda_graph_compatibility(cfg, args)
    model = _build_model(cfg, args.checkpoint, device)
    pvsa_timer = _attach_model_timer(model, cfg, args.debug_interval)
    height, width = args.input_size
    inputs = torch.randn(args.batch_size, 3, height, width, device=device)
    print(f"[COMPARE] model={cfg.model.backbone.get('model_name', 'compare')} "
          f"input=({height}, {width}) batch={args.batch_size} device={device}")

    try:
        runs = [
            _run_one(model, inputs, device, args, run_index)
            for run_index in range(1, args.repeat_times + 1)]
    finally:
        if pvsa_timer is not None:
            pvsa_timer.close()
    average_fps = sum(item["fps"] for item in runs) / len(runs)
    model_type = cfg.model.backbone.get("type", "unknown")
    result = dict(
        model_type=model_type,
        model_family="pvsa" if _is_pvsa_model(cfg) else "compare",
        config=os.path.abspath(args.config),
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else None,
        input_size=[height, width],
        batch_size=args.batch_size,
        debug=args.debug,
        cuda_graph=args.cuda_graph,
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
