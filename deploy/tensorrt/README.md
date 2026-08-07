# PVSA TensorRT 部署插件

本目录提供 PVSA Top-p 路由和 Flash Attention 的 TensorRT 插件化实现。
插件复用了当前项目中的 CUDA kernel 逻辑，但不依赖 PyTorch 运行时。

## 当前支持范围

第一版插件为了保证和原始 CUDA kernel 的数值一致，固定使用：

```text
输入布局：NHWC/窗口展平布局，必须与 PVSA 内部张量布局一致
数据类型：FP32
窗口大小：n_win=7
窗口数量：p2=49
head_dim=32
num_heads：2、4、8 或 16
topk：1 到 49
H、W：必须是 7 的倍数
```

插件由两个算子组成：

```text
PVSA_TopP_Route
    query、key
    -> route_weight、route_idx、keep_len

PVSA_TopP_Flash
    q_pix、kv_pix、route_weight、route_idx、keep_len
    -> attention_output
```

当前版本是固定形状、FP32 的第一版部署接口。FP16、INT8、动态路由上限和融合路由插件需要在数值校验通过后再扩展。

## 编译

需要安装与目标显卡匹配的 CUDA 和 TensorRT，并设置 TensorRT 根目录：

```bash
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/usr/local/TensorRT
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build/tensorrt -j$(nproc)
```

默认不启用 `--use_fast_math`，便于先做数值一致性验证。确认误差和分割指标满足要求后，
可以重新配置并打开快速数学选项：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPVSA_TRT_FAST_MATH=ON
cmake --build build/tensorrt_fast -j$(nproc)
```

编译产物：

```text
build/tensorrt/libpvsa_tensorrt_plugins.so
build/tensorrt/pvsa_build_plugin_engine
```

## 构建插件冒烟引擎

仓库同时提供一个最小网络构建程序。它把两个插件串成：

```text
query、key -> PVSA_TopP_Route
q_pix、kv_pix、路由输出 -> PVSA_TopP_Flash
```

这个程序不包含主干的卷积、线性层和解码头，作用是先验证插件注册、输入输出形状、
引擎序列化以及运行时加载流程：

```bash
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

构建完成后，部署程序必须在反序列化引擎前加载：

```text
libpvsa_tensorrt_plugins.so
```

完整 PVSA 网络仍需要把原模型中的 QKV 投影、窗口重排、路由插件和输出投影接入同一
TensorRT 网络。推荐先用上面的冒烟引擎确认环境，再通过 TensorRT 网络定义接口添加
插件层；不要把 `.cu` 文件直接交给 TensorRT，也不要在插件的 `enqueue` 中调用
PyTorch 接口。

`CMAKE_CUDA_ARCHITECTURES` 必须按实际显卡修改，例如：

```text
75：Turing
86：Ampere
89：Ada
90：Hopper
```

## 接入方式

构建 TensorRT 网络时，需要先加载插件库：

```cpp
initLibNvInferPlugins(&logger, "");
```

然后通过 TensorRT 插件注册表创建：

```text
PVSA_TopP_Route
PVSA_TopP_Flash
```

如果从 ONNX 构建网络，需要把 PVSA 的自定义算子导出为对应的自定义节点，并确保节点名称与插件名称一致。也可以使用 TensorRT 网络定义接口直接添加插件层。

## 推荐接入顺序

1. 先固定输入尺寸和 batch=1；
2. 先使用 FP32 验证 TensorRT 与 PyTorch 输出；
3. 验证 `max_abs_error`、`mean_abs_error` 和分割指标；
4. 再增加 FP16 kernel；
5. 最后再处理动态输入、INT8 和融合插件。

## 重要注意事项

- 当前路由 kernel 要求 `query` 和 `key` 为 FP32；
- 当前 Flash kernel 输出为 FP32，即使后续增加 FP16 输入，也要检查后续 `wo` 层的数据类型转换；
- `keep_len` 的数值虽然动态，但张量形状固定为 `[N,49]`，第一版可以通过固定 `topk` 支持；`q_len` 必须等于 `(H/7)*(W/7)`，`kv_len` 可以由窗口下采样结果决定；
- PVSA 的自动补齐和最终裁剪必须在 TensorRT 图中保留；
- 插件的 `enqueue` 不能执行主机同步、动态设备内存申请或不可捕获的 CUDA API，否则不能安全使用 CUDA Graph；
- TensorRT engine 加载时必须同时加载 `libpvsa_tensorrt_plugins.so`；
- 当前工作区没有 TensorRT 头文件和 CUDA 编译环境，因此 Windows 环境只能完成源代码和静态检查，实际编译必须在 Linux+CUDA+TensorRT 环境进行。

## 文件说明

```text
include/pvsa_topp_kernel.cuh       原始指针 CUDA launcher 接口
src/pvsa_topp_kernel.cu             不依赖 PyTorch 的 CUDA kernel 和 launcher
include/pvsa_topp_plugins.h         TensorRT 插件类声明
src/pvsa_topp_plugins.cpp           TensorRT 插件实现和注册
CMakeLists.txt                      插件动态库和冒烟引擎编译配置
tools/build_plugin_engine.cpp        固定形状插件冒烟引擎构建程序
```
