# PVSA-Net v3.0

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/work_dirs/PVSA
```

## 推理并保存分割结果

显示推理结果（分割结果图保存到 `--show-dir` 指定目录）：

### 原始路径推理（torch）

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v3.0:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/vis_results/gqy \
  --cfg-options model.backbone.topp_flash_backend=None \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 自定义 CUDA 核推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v3.0:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/vis_results/2 \
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
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v3.0:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v3.0/vis_results/3 \
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

## TensorRT 部署

当前版本提供 PVSA Top-p 路由和 Flash Attention 的 TensorRT 插件化实现。

完整的环境准备、插件编译、冒烟引擎构建、运行时加载、完整网络接入和数值验证说明见：

```text
deploy_readme.md
```

插件源码级接口和编译配置见：

```text
deploy/tensorrt/
```
