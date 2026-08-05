"""原始 PVSA 方法的统一公平基准测速入口。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from mmengine import Config
from mmseg.utils import register_all_modules

from benchmark_compare import (_build_model, _run_one, parse_args)
from pvsa_fair_timer import PVSAFairStageTimer


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.iters <= 0:
        raise ValueError("--batch-size 和 --iters 必须为正数")
    if args.debug_interval <= 0:
        raise ValueError("--debug-interval 必须为正数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("未检测到显卡，公平基准测速需要显卡设备")
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
    model = _build_model(cfg, args.checkpoint, device)
    backbone = model.backbone
    timer = PVSAFairStageTimer(
        backbone,
        model_name=cfg.model.backbone.get("model_name", "pvsa"),
        interval=args.debug_interval)
    # 复用统一脚本的计时接口，不改变原始 PVSA 主干代码。
    backbone.compare_timer = timer
    backbone.set_attention_debug = timer.configure
    backbone.flush_attention_report = timer.flush

    height, width = args.input_size
    inputs = torch.randn(args.batch_size, 3, height, width, device=device)
    print(f"[COMPARE] model={cfg.model.backbone.get('model_name', 'pvsa')} "
          f"input=({height}, {width}) batch={args.batch_size} device={device}")
    try:
        runs = [
            _run_one(model, inputs, device, args, run_index)
            for run_index in range(1, args.repeat_times + 1)]
    finally:
        timer.close()

    average_fps = sum(item["fps"] for item in runs) / len(runs)
    result = dict(
        method="pvsa_fair",
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
