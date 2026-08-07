# PVSA-Net v3.0 部署说明

本文档是 PVSA-Net v3.0 的独立部署说明，面向需要使用 CUDA 内核和 TensorRT 插件进行推理部署的场景。

当前仓库提供的是 **PVSA Top-p 路由和 Flash Attention 的 TensorRT 插件实现**。插件已经从 PyTorch 扩展中拆出，不依赖 PyTorch 运行时；但完整的 PVSA 分割网络仍需要将 QKV 投影、窗口重排、插件层、输出投影和解码头接入同一个 TensorRT 网络。

---

## 一、部署目录

```text
deploy/tensorrt/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── pvsa_topp_kernel.cuh
│   └── pvsa_topp_plugins.h
├── src/
│   ├── pvsa_topp_kernel.cu
│   └── pvsa_topp_plugins.cpp
└── tools/
    └── build_plugin_engine.cpp
```

文件作用：

| 文件 | 作用 |
|---|---|
| `pvsa_topp_kernel.cu` | 不依赖 PyTorch 的 CUDA 路由和注意力内核 |
| `pvsa_topp_plugins.cpp` | TensorRT 插件实现和插件注册 |
| `pvsa_topp_plugins.h` | 插件类声明 |
| `pvsa_topp_kernel.cuh` | CUDA launcher 接口 |
| `build_plugin_engine.cpp` | 构建固定形状插件冒烟引擎 |
| `CMakeLists.txt` | 编译动态库和冒烟引擎 |
| `deploy/tensorrt/README.md` | 插件源码级说明 |

原始 PyTorch 训练和推理路径仍保留在：

```text
mmseg/ops/topp_flash/
mmseg/models/utils/topp_flash_kernel.py
```

TensorRT 部署代码不会覆盖原始训练路径。

---

## 二、当前支持范围

第一版部署插件采用固定 FP32 接口，支持：

```text
数据类型：FP32
窗口大小：n_win=7
窗口数量：p2=49
每头通道数：head_dim=32
num_heads：2、4、8、16
qk_dim：64、128、256、512
qk_dim == dim
topk：1 到 49
H、W：必须是 7 的倍数
```

其中：

```text
q_len = (H / 7) * (W / 7)
```

`q_len` 必须满足上述关系；`kv_len` 可以根据窗口下采样结果变化，但必须大于 0。

当前版本暂不承诺：

```text
FP16
INT8
动态输入尺寸
动态 topk
融合路由插件
完整 PVSA 网络自动导出
```

建议先完成 FP32 数值一致性验证，再扩展 FP16、INT8 和动态形状。

---

## 三、准备部署环境

以下命令在 Linux 项目根目录执行：

```bash
cd /path/to/PVSA-v3.0

export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/usr/local/TensorRT
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

需要准备：

```text
CUDA Toolkit
TensorRT 开发包
CMake >= 3.18
GCC/G++ 与 CUDA 版本兼容
```

检查环境：

```bash
nvcc --version
cmake --version
echo "TENSORRT_ROOT=${TENSORRT_ROOT}"
test -f "${TENSORRT_ROOT}/include/NvInfer.h"
test -f "${TENSORRT_ROOT}/lib/libnvinfer.so" || test -f "${TENSORRT_ROOT}/lib64/libnvinfer.so"
```

如果上述 `test` 报错，说明 `TENSORRT_ROOT` 不是实际的 TensorRT 开发包目录。可以先查找头文件和库文件：

```bash
find /usr/local /opt /usr -type f \
  \( -name NvInfer.h -o -name "libnvinfer.so*" \) 2>/dev/null
```

正确的 TensorRT 根目录应当满足：

```text
<TENSORRT_ROOT>/include/NvInfer.h
<TENSORRT_ROOT>/lib/libnvinfer.so   或   <TENSORRT_ROOT>/lib64/libnvinfer.so
```

如果 TensorRT 安装在 Debian/Ubuntu 的系统目录，也可以直接传入两个路径：

```bash
cmake -S deploy/tensorrt -B build/tensorrt \
  -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DTENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

如果显卡不是 `Ampere`，需要按照实际显卡修改 `CMAKE_CUDA_ARCHITECTURES`：

```text
75：Turing
86：Ampere
89：Ada
90：Hopper
```

---

## 四、编译 TensorRT 插件

### 4.1 编译数值一致性版本

默认不启用 `--use_fast_math`，用于优先验证 TensorRT 输出与 PyTorch/CUDA 输出的一致性：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build/tensorrt -j$(nproc)
```

注意：Bash 的多行命令续行符必须是**一个反斜杠** `\`，并且反斜杠后面不能再有空格或第二个反斜杠。如果不想使用多行写法，也可以直接执行：

```bash
cmake -S deploy/tensorrt -B build/tensorrt -DTENSORRT_ROOT="$TENSORRT_ROOT" -DCMAKE_CUDA_ARCHITECTURES=86
```

编译产物：

```text
build/tensorrt/libpvsa_tensorrt_plugins.so
build/tensorrt/pvsa_build_plugin_engine
```

### 4.2 编译性能版本

确认数值误差和分割指标满足要求后，可以启用快速数学优化：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt_fast \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPVSA_TRT_FAST_MATH=ON

cmake --build build/tensorrt_fast -j$(nproc)
```

性能版本必须重新进行数值验证，不能直接认为与 FP32 版本完全一致。

---

## 五、插件接口

### 5.1 `PVSA_TopP_Route`

输入：

```text
query [N, 49, qk_dim]   FP32
key   [N, 49, qk_dim]   FP32
```

输出：

```text
route_weight [N, 49, topk]   FP32
route_idx    [N, 49, topk]   INT32
keep_len     [N, 49]         INT32
```

插件参数：

```text
topk
p
temperature
energy
scale
full_route
```

`full_route=true` 时必须使用 `topk=49`。

### 5.2 `PVSA_TopP_Flash`

输入：

```text
q_pix        [N, 49, q_len, qk_dim]       FP32
kv_pix       [N, 49, kv_len, qk_dim+dim]   FP32
route_weight [N, 49, topk]                FP32
route_idx    [N, 49, topk]                INT32
keep_len     [N, 49]                      INT32
```

输出：

```text
attention_output [N, H, W, dim]           FP32
```

输出布局为 NHWC。后续输出投影、跨阶段融合、裁剪和解码头必须按照原始 PVSA 网络的布局要求连接。

---

## 六、构建插件冒烟引擎

仓库提供一个最小网络构建程序，用来检查 TensorRT 是否能够正确发现并连接两个插件：

```text
query、key
    │
    ▼
PVSA_TopP_Route
    │
    ├── route_weight
    ├── route_idx
    └── keep_len
             │
q_pix、kv_pix ─┴─> PVSA_TopP_Flash
                         │
                         ▼
                 attention_output
```

构建固定形状测试引擎：

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

构建成功后应生成：

```text
work_dirs/pvsa_plugin_smoke.engine
```

该冒烟引擎只验证插件子图，不包含完整的卷积、线性层、特征融合和分割解码头。

---

## 七、TensorRT 运行时加载

反序列化引擎前，必须保证插件动态库已经加载。

### 7.1 C++ 部署程序

可以在部署程序启动时加载插件动态库：

```cpp
#include <dlfcn.h>

void* handle = dlopen(
    "./libpvsa_tensorrt_plugins.so",
    RTLD_NOW | RTLD_GLOBAL);
if (handle == nullptr) {
    throw std::runtime_error(dlerror());
}
```

然后初始化 TensorRT 插件注册表：

```cpp
initLibNvInferPlugins(&logger, "");
```

创建插件层时使用以下名称：

```text
PVSA_TopP_Route
PVSA_TopP_Flash
```

程序退出前不要提前卸载插件动态库，否则 TensorRT 运行时仍可能找不到插件实现。

### 7.2 使用 `trtexec` 检查引擎

```bash
trtexec \
  --loadEngine=work_dirs/pvsa_plugin_smoke.engine \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --warmUp=200 \
  --iterations=1000
```

如果提示找不到插件，需要先设置动态库搜索路径：

```bash
export LD_LIBRARY_PATH=$PWD/build/tensorrt:$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

---

## 八、接入完整 PVSA 网络

完整接入建议按照以下顺序进行：

### 第一步：保留原始前端计算

以下部分可以继续由 TensorRT 原生层实现：

```text
输入预处理
卷积层
QKV 线性投影
窗口划分
窗口内 Q/K/V 重排
KV 下采样
```

### 第二步：整理路由输入

将路由池化后的 Q、K 整理为：

```text
query [N,49,qk_dim]
key   [N,49,qk_dim]
```

然后添加 `PVSA_TopP_Route` 插件层。

### 第三步：整理注意力输入

将 Q 和 KV 整理为：

```text
q_pix  [N,49,q_len,qk_dim]
kv_pix [N,49,kv_len,qk_dim+dim]
```

并将路由插件的三个输出直接连接到 `PVSA_TopP_Flash`：

```text
route_weight -> route_weight
route_idx    -> route_idx
keep_len     -> keep_len
```

### 第四步：连接输出投影

Flash 插件输出为：

```text
[N,H,W,dim]
```

需要继续连接：

```text
lepe/局部增强
输出线性层 wo
自动补齐区域裁剪
跨阶段融合
解码头
```

自动补齐时，TensorRT 图中必须保留最终裁剪逻辑，不能只在主机端裁剪，否则输出尺寸和原始模型不一致。

---

## 九、数值一致性验证

建议至少比较以下结果：

```text
route_weight
route_idx
keep_len
attention_output
最终分割结果
```

推荐记录：

```text
max_abs_error
mean_abs_error
相对误差
Dice
IoU
FPS
```

验证流程：

1. 使用相同的随机输入或真实输入；
2. 使用原始 PyTorch/CUDA 路径计算参考输出；
3. 使用 TensorRT 插件路径计算部署输出；
4. 对比插件输出误差；
5. 再比较完整分割结果和指标；
6. 最后再测试 CUDA Graph、FP16 和性能优化。

注意：不能只比较最终 FPS。阶段输出误差、路由索引和最终分割指标都必须检查。

---

## 十、常见问题

### 10.1 找不到 `NvInfer.h`

检查：

```bash
ls $TENSORRT_ROOT/include/NvInfer.h
```

重新配置：

```bash
cmake -S deploy/tensorrt \
  -B build/tensorrt \
  -DTENSORRT_ROOT=$TENSORRT_ROOT
```

### 10.2 找不到 `libnvinfer.so`

设置：

```bash
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

### 10.3 找不到 `PVSA_TopP_Route` 或 `PVSA_TopP_Flash`

确认：

```bash
ls build/tensorrt/libpvsa_tensorrt_plugins.so
```

并确保在创建网络或反序列化引擎前加载该动态库。

### 10.4 提示输入形状不支持

检查：

```text
H、W 是否能被 7 整除
p2 是否为 49
q_len 是否等于 (H/7)*(W/7)
qk_dim 是否等于 dim
qk_dim / num_heads 是否等于 32
topk 是否在 1 到 49 之间
```

### 10.5 TensorRT 可以构建，但运行时返回错误

优先检查：

```text
TensorRT 运行时版本是否与构建版本一致
CUDA 架构是否包含当前显卡
插件动态库是否在进程退出前保持加载
输入张量是否为连续 FP32
route_idx 和 keep_len 是否为 INT32
```

### 10.6 Windows 环境无法编译

当前部署插件使用 CUDA 和 TensorRT Linux 开发环境进行验证。Windows 工作机可以进行源码检查，但不能替代 Linux 环境完成最终编译和运行验证。

---

## 十一、当前限制和后续计划

当前版本的目标是完成 CUDA 内核到 TensorRT 插件的第一阶段迁移，优先保证：

```text
插件可以注册
插件可以加入 TensorRT 网络
插件可以序列化和反序列化
固定 FP32 输入下可以运行
输出可以与原始 CUDA 路径比较
```

后续可以继续增加：

```text
FP16 输入和输出
动态 batch
动态输入尺寸
INT8 校准
ONNX 自定义节点导出
完整 PVSA 分割引擎构建脚本
TensorRT 与 PyTorch 自动对比工具
```

---

## 十二、相关文件

```text
README.md
deploy_readme.md
deploy/tensorrt/README.md
deploy/tensorrt/CMakeLists.txt
deploy/tensorrt/src/pvsa_topp_kernel.cu
deploy/tensorrt/src/pvsa_topp_plugins.cpp
deploy/tensorrt/tools/build_plugin_engine.cpp
```
