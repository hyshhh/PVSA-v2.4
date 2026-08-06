# 对比实验说明

## 一、统一环境变量

以下命令在项目根目录执行。权重和结果目录可以按实际路径修改。本说明中的公平测速命令均不加载训练好的权重，使用随机初始化模型。

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v2.4
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

统一测速脚本支持以下两种模式：

- `--cuda-graph true`：使用固定输入捕获并重放 CUDA Graph，统计整体吞吐率。本说明中的测速命令统一使用该模式；
- `--cuda-graph false`：仅用于 CUDA 后端未编译完成或需要排查问题时的普通前向测试。

开启 `--debug` 时，程序会在普通前向阶段输出注意力耗时和阶段总耗时。若同时使用 `--cuda-graph true`，整体 `fps` 来自 CUDA Graph，阶段耗时来自随后自动执行的普通前向调试阶段。

## 二、BiFormer-T 的 CUDA Graph 命令

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/compare/benchmark_biformer_t.py \
  configs-h/compare/biformer_t-compare_gqy-256x256.py \
  --input-size 256 256 \
  --cuda-graph true \
  --cudnn-benchmark \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --output work_dirs/compare/biformer_t/fps_attention_cuda_graph.json
```

## 三、PVSA 公平基准测速

### PVSA 普通前向公平测速

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/compare/benchmark_pvsa.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options \
  model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=0 \
  --input-size 256 256 \
  --cuda-graph true \
  --cudnn-benchmark \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --output work_dirs/compare/pvsa/fps_attention_cuda_graph.json
```

PVSA 的统一公平基准使用 `tools/analysis_tools/compare/pvsa_fair_timer.py` 统计 `PA` 注意力模块和阶段外层总耗时；原始测速入口仍然保留，不影响原有测试方法。结果表中的 PVSA 阶段数据应统一来自该公平基准，不要与原始入口的旧版调试输出混用。

本说明统一使用 `--cuda-graph true`。PVSA 必须设置 `model.backbone.topp_flash_backend=cuda`，并确认自定义 CUDA 扩展已经编译完成；如果扩展不可用，才改用 `--cuda-graph false`。

## 四、BiFormer-T，支持 S、B（更换字母）

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/biformer_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/biformer_t
```

### 普通前向测速与阶段耗时命令

```bash
python tools/analysis_tools/compare/benchmark_biformer_t.py \
  configs-h/compare/biformer_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --cuda-graph true \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_t/fps_attention.json
```

测试 BiFormer-S、BiFormer-B 时，将配置文件、测速脚本和结果目录中的 `t` 分别替换为 `s`、`b`。

## 五、Swin-T，支持 S、B（更换字母）

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/swin_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/swin_t
```

### 普通前向测速与阶段耗时命令

```bash
python tools/analysis_tools/compare/benchmark_swin_t.py \
  configs-h/compare/swin_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --cuda-graph true \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_t/fps_attention.json
```

测试 Swin-S、Swin-B 时，将配置文件、测速脚本和结果目录中的 `t` 分别替换为 `s`、`b`。

## 六、ViT-T，支持 S、B（更换字母）

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/vit_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/vit_t
```

### 普通前向测速与阶段耗时命令

```bash
python tools/analysis_tools/compare/benchmark_vit_t.py \
  configs-h/compare/vit_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --cuda-graph true \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_t/fps_attention.json
```

测试 ViT-S、ViT-B 时，将配置文件、测速脚本和结果目录中的 `t` 分别替换为 `s`、`b`。

## 七、输出示例与结果记录

阶段调试输出示例：

```text
[COMPARE-ATTN] model=swin_tiny images=100 S1_attention=5.5716ms S1_total=8.2031ms S2_attention=7.4840ms S2_total=10.6420ms S3_attention=11.3162ms S3_total=15.0245ms S4_attention=1.2586ms S4_total=2.1137ms
```

其中：

- `S1`、`S2`、`S3`、`S4`：对应阶段所有注意力模块的平均耗时；
- `S1_total`、`S2_total`、`S3_total`、`S4_total`：对应阶段外层总耗时；
- 对比模型的阶段总耗时包含卷积下采样和 Transformer Block；
- PVSA 的阶段总耗时包含 CNN 分支、Transformer 下采样、Transformer Block 和 FAM，不包含后置跨阶段融合、输出归一化与解码头；
- `fps`：完整分割模型前向吞吐率，包含解码头。

| 模型 | FPS | S1 注意力毫秒 | S1 总耗时毫秒 | S2 注意力毫秒 | S2 总耗时毫秒 | S3 注意力毫秒 | S3 总耗时毫秒 | S4 注意力毫秒 | S4 总耗时毫秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PVSA |  |  |  |  |  |  |  |  |  |
| BiFormer-T |  |  |  |  |  |  |  |  |  |
| BiFormer-S |  |  |  |  |  |  |  |  |  |
| BiFormer-B |  |  |  |  |  |  |  |  |  |
| Swin-T |  |  |  |  |  |  |  |  |  |
| Swin-S |  |  |  |  |  |  |  |  |  |
| Swin-B |  |  |  |  |  |  |  |  |  |
| ViT-T |  |  |  |  |  |  |  |  |  |
| ViT-S |  |  |  |  |  |  |  |  |  |
| ViT-B |  |  |  |  |  |  |  |  |  |

## 八、相关文件

- 原始 PVSA 测速：`tools/analysis_tools/benchmark.py`
- PVSA 公平基准测速：`tools/analysis_tools/compare/benchmark_pvsa.py`
- PVSA 公平计时器：`tools/analysis_tools/compare/pvsa_fair_timer.py`
- 统一对比测速入口：`tools/analysis_tools/compare/benchmark_compare.py`
- 对比模型代码：`mmseg/models/backbones/compare/`
- 对比实验配置：`configs-h/compare/`
