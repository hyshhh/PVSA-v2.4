# 对比实验说明

本目录按照 `plan.md` 单独增加了九组对比实验：

| 模型 | 版本 | 注意力类型 | 配置文件 | 独立测速脚本 |
|---|---|---|---|---|
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



## 三、BiFormer-T

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
  work_dirs/compare/biformer_t/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_t/fps_attention.json
```
## 四、BiFormer-S

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
  work_dirs/compare/biformer_s/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_s/fps_attention.json
```

## 五、BiFormer-B

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
  work_dirs/compare/biformer_b/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/biformer_b/fps_attention.json
```

## 六、Swin-T

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
  work_dirs/compare/swin_t/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_t/fps_attention.json
```

## 七、Swin-S

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
  work_dirs/compare/swin_s/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_s/fps_attention.json
```

## 八、Swin-B

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
  work_dirs/compare/swin_b/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/swin_b/fps_attention.json
```

## 九、ViT-T

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
  work_dirs/compare/vit_t/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_t/fps_attention.json
```

## 十、ViT-S

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
  work_dirs/compare/vit_s/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_s/fps_attention.json
```

## 十一、ViT-B

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
  work_dirs/compare/vit_b/epoch_80.pth \
  --input-size 224 224 \
  --batch-size 1 \
  --warmup 30 \
  --iters 200 \
  --debug \
  --debug-interval 100 \
  --cudnn-benchmark \
  --output work_dirs/compare/vit_b/fps_attention.json
```

## 十二、输出示例与结果记录
| 模型 | FPS | S1 注意力毫秒 | S2 注意力毫秒 | S3 注意力毫秒 | S4 注意力毫秒 |
|---|---:|---:|---:|---:|---:|
| BiFormer-T |  |  |  |  |  |
| BiFormer-S |  |  |  |  |  |
| BiFormer-B |  |  |  |  |  |
| Swin-T |  |  |  |  |  |
| Swin-S |  |  |  |  |  |
| Swin-B |  |  |  |  |  |
| ViT-T |  |  |  |  |  |
| ViT-S |  |  |  |  |  |
| ViT-B |  |  |  |  |  |


