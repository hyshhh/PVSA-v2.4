# FPS 测速（benchmark.py）
## 1. 原始路径推理 fps（torch）打印 attention 各阶段耗时（与 CUDA 核对比）：
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
## 2. 自定义 CUDA 核推理 fps
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

## 3. CUDA Graph 推理（最高吞吐）
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 256 256 \
  --cuda-graph \
  --cudnn-benchmark \
  --batch-size 1
```

## 4. 复杂度统计

```bash
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```

## 5. `topp_flash_debug` 三档模式（PVSA 阶段耗时分析）

`model.backbone.topp_flash_debug` 支持三个档位：

| 值 | 含义 | 输出 |
|---|---|---|
| `0`（或 `false`） | 关闭计时 | 无 |
| `1` | 图级滚动 | 每 100 张图，S1..S4 每个 stage 各打印一行平均耗时 |
| `2` | 每个 PVSA-block | 每张图每个 block 单独打印一行（带序号 `S1[1]`、`S1[2]`...） |

> 兼容：旧写法 `true` 等价于 `2`，`false` 等价于 `0`。

- **S1..S3**（ToppAttention，CUDA 核）：行内子阶段 = `qkv / kv_down / lepe / router / attn / wo`，
  `Router kernel`、`Flash kernel` 是核内纯计算计时（与外层 `router`/`attn` 重复，加总时请忽略其中一个）。
- **S4**（plain Attention，普通全局注意力）：只有一个 `attn` 字段。
- 每行的 `x/q/kv/route` 是该 stage 的 shape，可据此判断哪个 stage 是瓶颈。

### 模式 1：每 100 张图打印 S1..S4 各一行

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=1 \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 模式 2：每个 PVSA-block 单独打印（定位慢 block）

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=2 \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 实验：整 block 单次计时（验证分阶段计时是否高估）

`PVSA_BLOCK_TOTAL_ONLY=1` 时（仅模式 1 生效），用一对 CUDA Event 包住整个
PVSA block（qkv→wo）只同步一次，把干净的整 block 耗时累进 `BLOCK_TOTAL` 列，
与分阶段计时同表输出。若 `BLOCK_TOTAL` 明显小于 `qkv+router+attn+wo+...` 合计，
说明分阶段计时的逐段 `synchronize()` 放大了测量值，实际耗时要少。

```bash
export PVSA_BLOCK_TOTAL_ONLY=1
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=1 \
  --input-size 224 224 \
  --cudnn-benchmark
```

> 说明：分阶段计时（模式 1/2）每段之间都 `synchronize()`，会打断 GPU 异步流水线，
> 各段加总常大于实际延迟。`BLOCK_TOTAL` 列就是用来量化这个高估幅度的对照。