# PVSA-Net v3.0 部署命令

## 1. 进入项目并设置环境

```bash
cd /media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-v3.0

export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
command -v g++-12 || sudo apt-get install gcc-12 g++-12
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## 2. 检查 CUDA、CMake 和 TensorRT

```bash
nvcc --version
cmake --version

echo "TENSORRT_ROOT=${TENSORRT_ROOT}"
ls -l /usr/include/x86_64-linux-gnu/NvInfer.h
ls -l /usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1
```

如果不知道 TensorRT 路径：

```bash
find /usr/local /opt /usr -type f \
  \( -name NvInfer.h -o -name "libnvinfer.so*" \) 2>/dev/null
```

## 3. 配置 TensorRT 插件

当前环境使用：

```text
TensorRT 头文件：/usr/include/x86_64-linux-gnu/NvInfer.h
TensorRT 库文件：/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1
CUDA 架构：86
```

删除旧缓存：

```bash
rm -rf build/tensorrt_fast
```

配置快速数学版本：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DPVSA_TRT_FAST_MATH=ON
```

配置数值一致性版本：

```bash
rm -rf build/tensorrt

cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.2.1 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
```

## 4. 编译

快速数学版本：

```bash
cmake --build build/tensorrt_fast -j$(nproc)
```

数值一致性版本：

```bash
cmake --build build/tensorrt -j$(nproc)
```

检查编译结果：

```bash
ls -lh build/tensorrt_fast/libpvsa_tensorrt_plugins.so
ls -lh build/tensorrt_fast/pvsa_build_plugin_engine
```

## 5. 构建插件测试引擎

```bash
mkdir -p work_dirs

build/tensorrt_fast/pvsa_build_plugin_engine \
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

检查引擎：

```bash
ls -lh work_dirs/pvsa_plugin_smoke.engine
```

## 6. 使用 `trtexec` 测试

```bash
export LD_LIBRARY_PATH=$PWD/build/tensorrt_fast:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

trtexec \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

## 7. TensorRT 插件接口

```text
PVSA_TopP_Route
输入：query [N,49,qk_dim]、key [N,49,qk_dim]
输出：route_weight、route_idx、keep_len

PVSA_TopP_Flash
输入：q_pix、kv_pix、route_weight、route_idx、keep_len
输出：attention_output [N,H,W,dim]
```

## 8. 输入限制

```text
数据类型：FP32
n_win：7
p2：49
head_dim：32
num_heads：2、4、8、16
qk_dim：64、128、256、512
qk_dim == dim
topk：1 到 49
H、W：必须能被 7 整除
q_len：(H/7)*(W/7)
```

## 9. 完整网络接入顺序

```text
输入
  -> 卷积/QKV 投影
  -> 窗口重排
  -> PVSA_TopP_Route
  -> PVSA_TopP_Flash
  -> 输出投影 wo
  -> 自动补齐区域裁剪
  -> 跨阶段融合
  -> 解码头
```

完整网络中需要连接：

```text
route_weight -> PVSA_TopP_Flash.route_weight
route_idx    -> PVSA_TopP_Flash.route_idx
keep_len     -> PVSA_TopP_Flash.keep_len
```

## 10. 数值验证

```bash
# 对比 route_weight、route_idx、keep_len 和 attention_output
# 再对比最终分割结果、Dice、IoU 和 FPS
```

验证顺序：

```text
FP32 数值一致性
  -> TensorRT 引擎运行
  -> 完整分割结果
  -> FPS
  -> FP16
  -> INT8
```

## 11. 常见问题

TensorRT 文件不存在：

```bash
find /usr/local /opt /usr -type f \
  \( -name NvInfer.h -o -name "libnvinfer.so*" \) 2>/dev/null
```

直接指定路径：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_INCLUDE_DIR=/实际/include路径 \
  -DTENSORRT_LIBRARY=/实际/libnvinfer.so路径 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DPVSA_TRT_FAST_MATH=ON
```

运行时找不到动态库：

```bash
export LD_LIBRARY_PATH=/实际/TensorRT/lib:$LD_LIBRARY_PATH
```

插件文件：

```text
deploy/tensorrt/CMakeLists.txt
deploy/tensorrt/src/pvsa_topp_kernel.cu
deploy/tensorrt/src/pvsa_topp_plugins.cpp
deploy/tensorrt/tools/build_plugin_engine.cpp
```
