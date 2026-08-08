# PVSA TensorRT 部署插件

## 1. 支持范围

本目录提供 `PVSA_TopP_Route` 和 `PVSA_TopP_Flash` 两个 TensorRT 插件，使用独立 CUDA 内核，不依赖 PyTorch 运行时。

```text
数据类型：FP32
输入布局：NHWC/窗口展平布局，需与 PVSA 内部布局一致
窗口大小：n_win=7
窗口数量：p2=49
head_dim=32
num_heads：2、4、8、16
topk：1 到 49
H、W：必须是 7 的倍数
```

## 2. 安装 CUDA 12 TensorRT

当前项目使用 CUDA 12.0，建议使用 TensorRT 8.6.1.6 的 CUDA 12.0 压缩包，不要使用 CUDA 13.x 版本的软件包。

```bash
export CUDACXX=/usr/bin/nvcc
export TRT_ROOT=$HOME/opt/TensorRT-8.6.1.6
export PATH=$TRT_ROOT/bin:/usr/bin:$PATH
export LD_LIBRARY_PATH=$TRT_ROOT/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

下载文件名：

```text
TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
```

解压：

```bash
mkdir -p "$HOME/opt"
tar -xzf ~/Downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz -C "$HOME/opt"
```

检查：

```bash
ls -lh "$TRT_ROOT/include/NvInfer.h"
ls -lh "$TRT_ROOT/lib/libnvinfer.so"
```

## 3. 编译普通版

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

普通版产物：

```text
build/tensorrt/libpvsa_tensorrt_plugins.so
build/tensorrt/pvsa_build_plugin_engine
```

## 4. 编译快速版

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

快速版产物：

```text
build/tensorrt_fast/libpvsa_tensorrt_plugins.so
build/tensorrt_fast/pvsa_build_plugin_engine
```

## 5. 构建测试引擎

普通版：

```bash
mkdir -p work_dirs
CUDA_VISIBLE_DEVICES=1 \
build/tensorrt/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke.engine \
  --batch 1 --num-heads 8 --qk-dim 256 --dim 256 \
  --height 56 --width 56 --kv-len 64 --topk 8
```

快速版：

```bash
CUDA_VISIBLE_DEVICES=1 \
build/tensorrt_fast/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke_fast.engine \
  --batch 1 --num-heads 8 --qk-dim 256 --dim 256 \
  --height 56 --width 56 --kv-len 64 --topk 8
```

## 6. 运行测试

```bash
CUDA_VISIBLE_DEVICES=1 \
"$TRT_ROOT/bin/trtexec" \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo --profilingVerbosity=detailed --warmUp=200 --iterations=1000
```

快速版将引擎替换为：

```text
work_dirs/pvsa_plugin_smoke_fast.engine
```

## 7. 接入接口

```text
PVSA_TopP_Route
输入：query、key
输出：route_weight、route_idx、keep_len

PVSA_TopP_Flash
输入：q_pix、kv_pix、route_weight、route_idx、keep_len
输出：attention_output
```

构建网络前先加载插件：

```cpp
initLibNvInferPlugins(&logger, "");
```

然后通过 TensorRT 插件注册表创建：

```text
PVSA_TopP_Route
PVSA_TopP_Flash
```

完整网络需要自行连接 QKV 投影、窗口重排、插件、输出投影和解码头。第一版为固定形状 FP32 接口，建议先验证数值一致性，再扩展 FP16、动态输入和 INT8。

## 8. 文件

```text
include/pvsa_topp_kernel.cuh
src/pvsa_topp_kernel.cu
include/pvsa_topp_plugins.h
src/pvsa_topp_plugins.cpp
tools/build_plugin_engine.cpp
CMakeLists.txt
```
