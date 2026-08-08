# PVSA-Net v3.0 部署命令

## 1. 设置环境

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0

export CUDACXX=/usr/bin/nvcc
export TRT_ROOT=$HOME/opt/TensorRT-8.6.1.6
export CUDNN_ROOT=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib
export PYTHONPATH=$PWD:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PATH=$TRT_ROOT/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN_ROOT:$TRT_ROOT/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

检查 CUDA：

```bash
nvcc --version
cmake --version
```

## 2. 检查 cuDNN 8

TensorRT 8.6.1 需要 `libcudnn.so.8`。当前 `openmmlab` 环境使用：

```bash
find "$CONDA_PREFIX" -type f -name 'libcudnn.so.8' -print
ls -lh "$CUDNN_ROOT/libcudnn.so.8"
ldd "$CUDNN_ROOT/libcudnn.so.8" | grep "not found"
```

如果路径不同，按 `find` 的结果修改 `CUDNN_ROOT`；找不到 `libcudnn.so.8` 时先安装 cuDNN 8。

## 3. 卸载 CUDA 13 TensorRT

只卸载已安装的 TensorRT 相关软件包，不要使用不存在的 `libnvparsers*` 软件包名：

```bash
dpkg-query -W -f='${binary:Package} ${db:Status-Status}\n' | awk '$2=="installed" && $1 ~ /^(tensorrt|libnvinfer|libnvonnxparsers|python3-libnvinfer)/ {print $1}' | xargs -r sudo apt-get purge -y
sudo apt-get autoremove -y
sudo ldconfig
```

## 4. 下载并解压 CUDA 12.0 对应的 TensorRT

TensorRT 压缩包需要从 NVIDIA 官方页面下载。登录后下载：

```text
TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
```

如果下载链接可直接访问：

```bash
wget -O /tmp/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz" && mkdir -p "$HOME/opt" && tar -xzf /tmp/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz -C "$HOME/opt"
```

如果 `wget` 返回 `403`，请在浏览器下载后执行：

```bash
mkdir -p "$HOME/opt"
tar -xzf ~/Downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz -C "$HOME/opt"
```

检查 TensorRT 文件：

```bash
ls -lh "$TRT_ROOT/include/NvInfer.h"
ls -lh "$TRT_ROOT/lib/libnvinfer.so"
ls -lh "$TRT_ROOT/bin/trtexec"
```

## 5. 编译普通版

普通版不启用快速数学，用于先验证数值一致性。

```bash
rm -rf build/tensorrt

cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR="$TRT_ROOT/include" \
  -DTENSORRT_LIBRARY="$TRT_ROOT/lib/libnvinfer.so" \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12

cmake --build build/tensorrt -j$(nproc)
```

产物：

```bash
ls -lh build/tensorrt/libpvsa_tensorrt_plugins.so
ls -lh build/tensorrt/pvsa_build_plugin_engine
```

## 6. 编译快速版

```bash
rm -rf build/tensorrt_fast

cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_INCLUDE_DIR="$TRT_ROOT/include" \
  -DTENSORRT_LIBRARY="$TRT_ROOT/lib/libnvinfer.so" \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DPVSA_TRT_FAST_MATH=ON

cmake --build build/tensorrt_fast -j$(nproc)
```

产物：

```bash
ls -lh build/tensorrt_fast/libpvsa_tensorrt_plugins.so
ls -lh build/tensorrt_fast/pvsa_build_plugin_engine
```

## 7. 构建插件测试引擎

`86` 对应 RTX A6000。GPU 2 当前负载较高，优先使用 GPU 1：

```bash
mkdir -p work_dirs

CUDA_VISIBLE_DEVICES=1 \
build/tensorrt/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke.engine \
  --batch 1 \
  --num-heads 8 \
  --qk-dim 256 \
  --dim 256 \
  --height 56 \
  --width 56 \
  --kv-len 64 \
  --topk 8
```

快速版：

```bash
CUDA_VISIBLE_DEVICES=1 \
build/tensorrt_fast/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke_fast.engine \
  --batch 1 \
  --num-heads 8 \
  --qk-dim 256 \
  --dim 256 \
  --height 56 \
  --width 56 \
  --kv-len 64 \
  --topk 8
```

## 8. 使用 `trtexec` 测试

普通版：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --plugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

快速版：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --plugins="$PWD/build/tensorrt_fast/libpvsa_tensorrt_plugins.so" \
  --loadEngine=work_dirs/pvsa_plugin_smoke_fast.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

运行包含自定义插件的引擎时，必须通过 `--plugins` 显式加载插件动态库：

```bash
--plugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so"
```

否则 `trtexec` 反序列化引擎时找不到 `PVSA_TopP_Route` 和 `PVSA_TopP_Flash`。

## 9. 插件接口与限制

```text
PVSA_TopP_Route
输入：query [N,49,qk_dim]、key [N,49,qk_dim]
输出：route_weight、route_idx、keep_len

PVSA_TopP_Flash
输入：q_pix、kv_pix、route_weight、route_idx、keep_len
输出：attention_output [N,H,W,dim]
```

```text
数据类型：FP32
n_win：7
p2：49
head_dim：32
num_heads：2、4、8、16
qk_dim == dim
H、W：必须能被 7 整除
topk：1 到 49
```

TensorRT 反序列化引擎前必须加载：

```text
$TRT_ROOT/lib/libnvinfer.so
build/tensorrt/libpvsa_tensorrt_plugins.so
```

插件源码：

```text
deploy/tensorrt/include/
deploy/tensorrt/src/
deploy/tensorrt/tools/build_plugin_engine.cpp
deploy/tensorrt/CMakeLists.txt
```

## 10. 完整 PVSA TensorRT 框架测速

完整部署的正确流程是：

```text
完整 PVSA 模型 -> 固定形状 ONNX -> TensorRT 完整引擎 -> trtexec 测速
```

完整引擎需要包含主干、PVSA 模块、输出投影和解码头，并且 ONNX 中的 PVSA 自定义节点必须映射到：

```text
PVSA_TopP_Route
PVSA_TopP_Flash
```

### 10.1 构建完整 TensorRT 引擎

以下命令以已经生成的完整模型文件为例：

```bash
export FULL_ONNX=work_dirs/pvsa_full.onnx
export FULL_ENGINE=work_dirs/pvsa_full.engine

CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --onnx="$FULL_ONNX" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --saveEngine="$FULL_ENGINE" \
  --verbose
```

固定输入尺寸的 ONNX 不需要额外设置动态形状；动态输入则必须补充对应的形状配置。
首次部署建议使用 FP32，验证数值一致性后再增加 `--fp16`。

### 10.2 完整 TensorRT 引擎测速

普通 TensorRT 推理：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine="$FULL_ENGINE" \
  --warmUp=500 \
  --duration=0 \
  --iterations=1000 \
  --useSpinWait
```

完整 TensorRT CUDA Graph 推理：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --staticPlugins="$PWD/build/tensorrt/libpvsa_tensorrt_plugins.so" \
  --loadEngine="$FULL_ENGINE" \
  --warmUp=500 \
  --duration=0 \
  --iterations=1000 \
  --useCudaGraph \
  --useSpinWait
```

确认日志包含：

```text
Plugins: .../libpvsa_tensorrt_plugins.so
CUDA Graph: Enabled
&&&& PASSED TensorRT.trtexec
```

### 10.3 当前仓库状态

```text
build/tensorrt/pvsa_build_plugin_engine
```

目前只构建 PVSA 插件冒烟引擎，不是完整 PVSA 网络引擎。
完整 ONNX 导出和完整网络 TensorRT 构建入口尚未集成到当前仓库，因此 `pvsa_full.onnx` 和 `pvsa_full.engine` 是完整网络转换完成后的文件名示例，不能用插件冒烟引擎代替。

PyTorch 完整框架测速属于算法性能测试，应使用 `FPS.md` 或 `tools/analysis_tools/benchmark.py`，不应作为 TensorRT 部署流程。
