# PVSA-Net

基于 MMSegmentation 的语义分割项目，当前分支只保留两条使用路径：

1. 原始路径：用于训练和普通推理。
2. 自定义 CUDA 核路径：只用于推理加速。
## 训练
只使用原始路径训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/gqy
```
如果训练中偶发出现超大梯度，可以临时打开梯度尖峰定位钩子，只在异常时打印梯度最大的参数：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 grad_spike_debug=True \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/nex/0.1111
```
## 原始路径推理
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.topp_flash_backend=None
```
## 复杂度统计
```bash
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```
按 stage 看 cnn / transformer / FAM / fusion：
```bash
python tools/analysis_tools/pvsa_stage_complexity.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```
## 推理并保存分割结果
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/gqy
```
## 自定义 CUDA 核推理
首次运行或修改 CUDA 源码后，建议先清理旧编译缓存：
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
```
TopP 推理只保留两个开关：
- `model.backbone.topp_flash_backend=None` 或 `cuda`
- `model.backbone.topp_flash_debug=False` 或 `True`

推理模板：
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=False
```

打印各环节时间时，只把 `model.backbone.topp_flash_debug` 改成 `True`。

最后一层默认使用 49 窗口全连接路由，不再需要额外开关。
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=True
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
