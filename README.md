# PVSA-Net

基于 MMSegmentation 的语义分割项目，当前分支只保留两条使用路径：

1. 原始路径：用于训练和普通推理。
2. 自定义 CUDA 核路径：只用于推理加速。

## 路径说明

### 原始路径

配置：

```text
model.backbone.use_topp_flash=False
```

用途：

- 训练
- 普通推理
- 作为自定义 CUDA 核结果的对照

路由权重补偿统一为：

```python
topk_score = topk_score * energy
```

其中 `energy` 仍从配置文件读取，可以继续调节。

### 自定义 CUDA 核推理路径

配置：

```text
model.backbone.use_topp_flash=True
model.backbone.topp_flash_backend=cuda
```

用途：

- 只用于推理
- 路由和注意力在融合 CUDA 核中完成
- 不再使用 `max_len` / `max_keep` 预扫描补偿
- 路由权重为 `prob * energy`

## 训练

只使用原始路径训练：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

## 原始路径推理

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=False \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

## 自定义 CUDA 核推理

首次运行或修改 CUDA 源码后，建议先清理旧编译缓存：

```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
```

推理命令：

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-V2.2:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend=cuda \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

如果需要查看编译日志：

```bash
export PVSA_TOPP_FLASH_VERBOSE=1
```

如果服务器 GPU 架构自动检测失败，可以手动指定：

```bash
export PVSA_TOPP_FLASH_ARCH="8.6"
```

## 注意事项

- 自定义 CUDA 核路径只面向推理，不用于训练。
- 真实速度测试时不要打开调试日志。
- 如果需要调整 `energy`、`p`、`temperature`、`maxk`，请修改配置文件中的 `topp_route_configs`。
- 本地不具备 CUDA 编译环境，CUDA 编译和性能验证以服务器结果为准。
