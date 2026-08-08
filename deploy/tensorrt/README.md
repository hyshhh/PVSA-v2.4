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

## 2. 编译普通版

```bash
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
command -v g++-12 || sudo apt-get install gcc-12 g++-12

rm -rf build/tensorrt
cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build/tensorrt -j$(nproc)
```

普通版产物：

```text
build/tensorrt/libpvsa_tensorrt_plugins.so
build/tensorrt/pvsa_build_plugin_engine
```

## 3. 编译快速版

```bash
rm -rf build/tensorrt_fast
cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1 \
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

## 4. 构建测试引擎

普通版：

```bash
mkdir -p work_dirs
build/tensorrt/pvsa_build_plugin_engine \
  --output work_dirs/pvsa_plugin_smoke.engine \
  --batch 1 --num-heads 8 --qk-dim 256 --dim 256 \
  --height 56 --width 56 --kv-len 64 --topk 8
```

快速版：将上面的可执行文件和输出文件分别替换为：

```text
build/tensorrt_fast/pvsa_build_plugin_engine
work_dirs/pvsa_plugin_smoke_fast.engine
```

## 5. 运行测试

普通版：

```bash
export LD_LIBRARY_PATH=$PWD/build/tensorrt:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
trtexec --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo --profilingVerbosity=detailed --warmUp=200 --iterations=1000
```

快速版：将动态库目录和引擎文件替换为：

```text
$PWD/build/tensorrt_fast
work_dirs/pvsa_plugin_smoke_fast.engine
```

## 6. 接入接口

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

## 7. 文件

```text
include/pvsa_topp_kernel.cuh
src/pvsa_topp_kernel.cu
include/pvsa_topp_plugins.h
src/pvsa_topp_plugins.cpp
tools/build_plugin_engine.cpp
CMakeLists.txt
```
