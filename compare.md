# 对比实验说明

本目录按照 `plan.md` 单独增加了十组对比实验；公平测速统一不加载训练好的权重：

| 模型 | 版本 | 注意力类型 | 配置文件 | 独立测速脚本 |
|---|---|---|---|---|
| PVSA | 原始模型 | Top-P 路由注意力 | `configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py` | `tools/analysis_tools/compare/benchmark_pvsa.py` |
| BiFormer | T | S1-S3 为 BRA，S4 为全局注意力 | `configs-h/compare/biformer_t-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_biformer_t.py` |
| BiFormer | S | S1-S3 为 BRA，S4 为全局注意力 | `configs-h/compare/biformer_s-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_biformer_s.py` |
| BiFormer | B | S1-S3 为 BRA，S4 为全局注意力 | `configs-h/compare/biformer_b-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_biformer_b.py` |
| Swin | T | 窗口注意力与移位窗口注意力 | `configs-h/compare/swin_t-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_swin_t.py` |
| Swin | S | 窗口注意力与移位窗口注意力 | `configs-h/compare/swin_s-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_swin_s.py` |
| Swin | B | 窗口注意力与移位窗口注意力 | `configs-h/compare/swin_b-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_swin_b.py` |
| ViT | T | 四个阶段均为全局自注意力 | `configs-h/compare/vit_t-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_vit_t.py` |
| ViT | S | 四个阶段均为全局自注意力 | `configs-h/compare/vit_s-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_vit_s.py` |
| ViT | B | 四个阶段均为全局自注意力 | `configs-h/compare/vit_b-compare_gqy-256x256.py` | `tools/analysis_tools/compare/benchmark_vit_b.py` |

所有对比主干均输出四个尺度的特征：`1/4`、`1/8`、`1/16`、`1/32`，可以直接接入相同的 `SegformerHead`。测速脚本统计完整分割网络的前向吞吐，同时由主干内部计时器统计 `S1` 到 `S4` 的注意力模块耗时。

## 一、统一环境变量

以下命令在项目根目录执行。权重和结果目录可以按实际路径修改。

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v2.4
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```



## 二、统一实验口径

- PVSA 和九个对比模型均使用固定输入尺寸测速。
- 默认批大小为一，预热三十次，正式测速二百次。
- 普通前向模式和 CUDA Graph 模式分别记录吞吐率。
- 阶段耗时统一按单张图片的平均注意力耗时统计。
- 开启 `--debug` 时，程序会额外进行普通前向阶段统计；CUDA Graph 本身不插入阶段事件计时。
- PVSA 公平基准使用前向钩子统计四个阶段的 `PA` 模块，原始 PVSA 测速入口仍然保留。
- 本文件中的公平测速命令不传入训练好的权重，统一使用随机初始化模型；训练命令只用于需要精度结果时的训练流程展示。
- 统一测速脚本的 `--cuda-graph` 默认值为 `true`，可以写成 `--cuda-graph true` 或 `--cuda-graph false`；需要普通前向测速时必须显式写 `--cuda-graph false`。

## 三、使用 CUDA Graph 测试统一对比脚本

统一测速脚本现在支持 `--cuda-graph`。它会使用固定输入尺寸捕获完整的主干和解码头前向，并通过重放图统计吞吐率。

如果同时添加 `--debug`，脚本会分成两个阶段：

1. 使用 CUDA Graph 测量整体吞吐率；
2. 关闭 CUDA Graph，使用普通前向单独统计 `S1` 到 `S4` 阶段注意力耗时。

因此最终结果中的 `fps` 是 CUDA Graph 吞吐率，而 `attention_reports` 来自普通前向调试阶段。这样既能得到图重放后的吞吐率，也能保留阶段耗时统计。

### BiFormer-T 的 CUDA Graph 命令

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

如果只测试 CUDA Graph 的纯吞吐率，可以保留默认值，或显式写：

```bash
--cuda-graph true
```

如果测试普通前向吞吐率，应显式写：

```bash
--cuda-graph false
```

阶段调试参数仍然可以按需删除：

```bash
--debug \
--debug-interval 100
```

此时不会执行第二阶段的注意力耗时统计。

注意事项：

- 必须使用固定输入尺寸；
- 建议批大小固定为一；
- 必须在显卡环境中运行；
- CUDA Graph 捕获期间不会执行阶段事件计时；
- 如果模型结构或输入尺寸发生变化，需要重新启动脚本；
- `--cuda-graph` 不需要设置 `model.backbone.topp_flash_backend=cuda`，因为这里使用的是对比实验专用主干。

其他模型只需要将脚本、配置文件和输出文件名替换为对应版本即可。

## 四、PVSA 公平基准测速

原始 PVSA 方法现在增加了统一公平基准入口：

```text
tools/analysis_tools/compare/benchmark_pvsa.py
```

该入口与其他对比模型保持相同测速口径：

- 固定输入尺寸；
- 相同批大小；
- 相同预热次数；
- 相同正式迭代次数；
- 相同显卡同步方式；
- 相同 `S1` 到 `S4` 阶段注意力统计方式；
- 支持 CUDA Graph 吞吐率测试。

PVSA 公平基准计时通过前向钩子统计原始 PVSA 四个阶段中的 `PA` 模块，不修改原始 PVSA 主干和原始测速脚本。

### PVSA 普通前向公平测速

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/compare/benchmark_pvsa.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options \
  model.backbone.topp_flash_backend=None \
  model.backbone.topp_flash_debug=0 \
  --input-size 256 256 \
  --cuda-graph false \
  --cudnn-benchmark \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --output work_dirs/compare/pvsa/fps_attention_eager.json
```

### PVSA CUDA Graph 公平测速

如果要和其他对比模型一样使用 CUDA Graph，需要启用 PVSA 的自定义 CUDA 后端：

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

其中 CUDA Graph 模式下：

- `fps` 是图捕获与重放得到的吞吐率；
- `attention_reports` 是随后普通前向阶段统计得到的 `S1` 到 `S4` 注意力耗时；
- 若自定义 CUDA 扩展没有编译成功，先使用普通前向公平测速，或按照原始 PVSA 的 CUDA 扩展安装方式处理。

### 原始 PVSA 测试方法保留

原始测试入口没有删除或替换，仍然保留在：

```text
tools/analysis_tools/benchmark.py
```

原始入口的 `checkpoint` 参数是必填项，因此原始权重测试命令继续放在 `FPS.md` 中；本文件中的统一公平基准命令均不传入训练好的权重。

原始方法和统一公平基准方法的区别是：原始入口沿用 PVSA 原有数据读取、预热、计时和调试逻辑；公平基准入口使用固定随机输入、统一预热次数、统一迭代次数以及统一的阶段钩子计时。

## 五、BiFormer-T

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/biformer_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/biformer_t
```
### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_biformer_t.py \
  configs-h/compare/biformer_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_t/fps_attention.json
```
## 六、BiFormer-S

### 训练命令
```bash
python tools/train.py \
  configs-h/compare/biformer_s-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/biformer_s
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_biformer_s.py \
  configs-h/compare/biformer_s-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_s/fps_attention.json
```

## 七、BiFormer-B

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/biformer_b-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/biformer_b
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_biformer_b.py \
  configs-h/compare/biformer_b-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_b/fps_attention.json
```

## 八、Swin-T

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/swin_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/swin_t
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_swin_t.py \
  configs-h/compare/swin_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_t/fps_attention.json
```

## 九、Swin-S

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/swin_s-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/swin_s
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_swin_s.py \
  configs-h/compare/swin_s-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_s/fps_attention.json
```

## 十、Swin-B

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/swin_b-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/swin_b
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_swin_b.py \
  configs-h/compare/swin_b-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_b/fps_attention.json
```

## 十一、ViT-T

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/vit_t-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/vit_t
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_vit_t.py \
  configs-h/compare/vit_t-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_t/fps_attention.json
```

## 十二、ViT-S

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/vit_s-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/vit_s
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_vit_s.py \
  configs-h/compare/vit_s-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_s/fps_attention.json
```

## 十三、ViT-B

### 训练命令

```bash
python tools/train.py \
  configs-h/compare/vit_b-compare_gqy-256x256.py \
  --work-dir work_dirs/compare/vit_b
```

### FPS 与阶段注意力耗时命令

```bash
python tools/analysis_tools/compare/benchmark_vit_b.py \
  configs-h/compare/vit_b-compare_gqy-256x256.py \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_b/fps_attention.json
```

## 十四、输出示例与结果记录
| 模型 | FPS | S1 注意力毫秒 | S2 注意力毫秒 | S3 注意力毫秒 | S4 注意力毫秒 |
|---|---:|---:|---:|---:|---:|
| PVSA |  |  |  |  |  |
| BiFormer-T |  |  |  |  |  |
| BiFormer-S |  |  |  |  |  |
| BiFormer-B |  |  |  |  |  |
| Swin-T |  |  |  |  |  |
| Swin-S |  |  |  |  |  |
| Swin-B |  |  |  |  |  |
| ViT-T |  |  |  |  |  |
| ViT-S |  |  |  |  |  |
| ViT-B |  |  |  |  |  |



## 十五、文件归档

- 原始 PVSA 测速：`tools/analysis_tools/benchmark.py`
- PVSA 公平基准测速：`tools/analysis_tools/compare/benchmark_pvsa.py`
- PVSA 公平计时器：`tools/analysis_tools/compare/pvsa_fair_timer.py`
- 统一对比测速入口：`tools/analysis_tools/compare/benchmark_compare.py`
- 对比模型代码：`mmseg/models/backbones/compare/`
- 对比实验配置：`configs-h/compare/`
