# PVSA-Net v3.0 部署命令

## 1. 设置环境

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0

export CUDACXX=/usr/bin/nvcc
export TRT_ROOT=$HOME/opt/TensorRT-8.6.1.6
export PYTHONPATH=$PWD:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PATH=$TRT_ROOT/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=$TRT_ROOT/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

检查 CUDA：

```bash
nvcc --version
cmake --version
```

## 2. 卸载 CUDA 13 TensorRT

只卸载已安装的 TensorRT 相关软件包，不要使用不存在的 `libnvparsers*` 软件包名：

```bash
dpkg-query -W -f='${binary:Package} ${db:Status-Status}\n' | awk '$2=="installed" && $1 ~ /^(tensorrt|libnvinfer|libnvonnxparsers|python3-libnvinfer)/ {print $1}' | xargs -r sudo apt-get purge -y
sudo apt-get autoremove -y
sudo ldconfig
```

## 3. 下载并解压 CUDA 12.0 对应的 TensorRT

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

## 4. 编译普通版

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

## 5. 编译快速版

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

## 6. 构建插件测试引擎

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

## 7. 使用 `trtexec` 测试

普通版：

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
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
  --loadEngine=work_dirs/pvsa_plugin_smoke_fast.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

## 8. 插件接口与限制

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
