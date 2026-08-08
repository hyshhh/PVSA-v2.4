# PVSA-Net v3.0 部署命令

## 1. 环境

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0

export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
command -v g++-12 || sudo apt-get install gcc-12 g++-12
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## 2. 检查依赖

```bash
nvcc --version
cmake --version
ls -lh /usr/include/x86_64-linux-gnu/NvInfer.h
ls -lh /usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1
```

如果路径不同：

```bash
find /usr/local /opt /usr -type f \
  \( -name NvInfer.h -o -name "libnvinfer.so*" \) 2>/dev/null
```

## 3. 配置并编译普通版

普通版不启用快速数学，先用于数值一致性验证。

```bash
rm -rf build/tensorrt

cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12

cmake --build build/tensorrt -j$(nproc)
```

产物：

```bash
ls -lh build/tensorrt/libpvsa_tensorrt_plugins.so
ls -lh build/tensorrt/pvsa_build_plugin_engine
```

## 4. 配置并编译快速版

快速版启用 `--use_fast_math`，用于性能测试；产物与普通版分开保存。

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

产物：

```bash
ls -lh build/tensorrt_fast/libpvsa_tensorrt_plugins.so
ls -lh build/tensorrt_fast/pvsa_build_plugin_engine
```

## 5. 构建插件测试引擎

默认使用普通版：

```bash
mkdir -p work_dirs

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

快速版只需替换可执行文件：

```bash
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

## 6. 使用 `trtexec` 测试

普通版：

```bash
export LD_LIBRARY_PATH=$PWD/build/tensorrt:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

trtexec \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

快速版：

```bash
export LD_LIBRARY_PATH=$PWD/build/tensorrt_fast:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

trtexec \
  --loadEngine=work_dirs/pvsa_plugin_smoke_fast.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

## 7. 插件接口与限制

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

完整网络接入顺序：

```text
输入 -> QKV 投影 -> 窗口重排 -> PVSA_TopP_Route
     -> PVSA_TopP_Flash -> 输出投影 -> 裁剪 -> 解码头
```

TensorRT 反序列化引擎前必须加载：

```text
libpvsa_tensorrt_plugins.so
```

插件源码：

```text
deploy/tensorrt/include/
deploy/tensorrt/src/
deploy/tensorrt/tools/build_plugin_engine.cpp
deploy/tensorrt/CMakeLists.txt
```
