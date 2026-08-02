# PVSA-Net

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA
```

## 推理并保存分割结果

显示推理结果（分割结果图保存到 `--show-dir` 指定目录）：

### 原始路径推理（torch）

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/gqy \
  --cfg-options model.backbone.topp_flash_backend=None \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 自定义 CUDA 核推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/2 \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 自定义 CUDA 核推理 + CUDA Graph（最高吞吐）

在 `--cfg-options ... topp_flash_backend=cuda` 基础上加 `--cuda-graph`，
把模型 forward 捕获成 CUDA Graph 重放，推理可提速约 5 倍（结果与普通
predict 一致）。predict 后处理在图外执行但计入流程。

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/3 \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark \
  --cuda-graph
  
```

- 首次推理会打印 `CUDA Graph: 捕获完成`（含预热，稍慢），之后每张图重放。
- 若 `topp_flash_backend` 不是 cuda 会报错（torch 路径有动态路由形状，无法捕获）。

## FPS 测速

fps 测速、阶段耗时分析、CUDA Graph 高吞吐推理见 **`FPS.md`**。

## PVSA 阶段耗时分析（`topp_flash_debug`）

`model.backbone.topp_flash_debug` 三档：

| 值 | 含义 |
|---|---|
| `0`（或 `false`） | 关闭计时 |
| `1` | 每 100 张图，S1..S4 每个 stage 各一行平均耗时 |
| `2` | 每个 PVSA-block 单独一行（`S1[1]`、`S1[2]`...） |

用法示例（benchmark 测速时附加计时）：

```bash
--cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=2 \
  --input-size 224 224 \
  --cudnn-benchmark
```

详细说明与 `BLOCK_TOTAL` 整 block 对照实验见 **`FPS.md` 第 5 节**。
