# FPS 测速（benchmark.py）

推理并显示分割结果用 `README.md`，本文件只做 **fps 测速 / 阶段耗时分析 / 高吞吐推理**。

所有命令默认 **batch=1（真实环境 fps）**。

## 原始路径推理 fps（torch）

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=None \
  --input-size 224 224 \
  --cudnn-benchmark
```

打印 attention 各阶段耗时（与 CUDA 核对比）：

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=None \
  model.backbone.topp_flash_debug=true \
  --input-size 224 224 \
  --cudnn-benchmark
```

## 自定义 CUDA 核推理 fps

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark
```

## CUDA Graph 推理（最高吞吐）

把整个模型 forward 捕获成 CUDA Graph 并重放，消除每次推理的 kernel launch
与 Python 调度开销。**要求**：`topp_flash_backend=cuda` + 固定输入尺寸。

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark \
  --cuda-graph
```

- 首次会打印 `CUDA Graph: 捕获完成`（含预热，稍慢），之后每张图重放图。
- 若 `topp_flash_backend` 不是 cuda 会直接报错提示（torch 路径有动态路由形状，无法捕获）。
- 若捕获失败（如 CUDA 内存、形状变化），会报错而不是静默回退。

## 纯模型 forward 测速（跳过 predict 后处理）

对比框架后处理开销：`--raw` 只测 backbone+decode_head 的纯 forward。

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  --input-size 224 224 \
  --cudnn-benchmark \
  --raw
```

## 复杂度统计

```bash
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```

## 通用参数

| 参数 | 作用 |
|---|---|
| `--input-size H W` | 调整测试输入分辨率（覆盖 Resize pipeline 与 data_preprocessor.size）。尺寸需能被 `n_win=7` 整除（如 224/256/448/512），CUDA 核路径要求整除否则自动回退 torch |
| `--cudnn-benchmark` | 开启 cuDNN 自动调优。固定输入尺寸时通常快 5-15%；首次推理有 autotune 预热 |
| `--batch-size N` | 默认 1（真实环境单图 fps）。增大可测吞吐，但单图延迟不变 |
| `--raw` | 纯模型 forward（跳过 predict 后处理），诊断框架开销 |
| `--cuda-graph` | CUDA Graph 捕获重放（最高吞吐），需 `topp_flash_backend=cuda` |

## 正确性说明（自定义核 vs 原始路径）

- 自定义 CUDA 核的 `use_route_weight` 自动跟随配置的 `soft_routing`：
  `soft_routing=False` 时核内不乘 route_weight，与原始路径 KVGather 行为一致。
- 修改 `soft_routing` 等路由行为需改配置文件并重训。
