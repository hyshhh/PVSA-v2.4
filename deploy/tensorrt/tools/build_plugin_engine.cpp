#include <NvInfer.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << (message == nullptr ? "" : message)
                << '\n';
    }
  }
};

struct Options {
  std::string output = "pvsa_plugin_smoke.engine";
  int32_t batch = 1;
  int32_t num_heads = 8;
  int32_t qk_dim = 256;
  int32_t dim = 256;
  int32_t n_win = 7;
  int32_t height = 56;
  int32_t width = 56;
  int32_t kv_len = 64;
  int32_t topk = 8;
  float route_p = 0.9f;
  float route_temperature = 1.0f;
  float route_energy = 1.0f;
  float route_scale = 1.0f;
  float flash_scale = 1.0f;
  bool use_route_weight = true;
};

void print_usage(const char* program) {
  std::cout
      << "用法：" << program << " [选项]\n\n"
      << "该程序构建一个固定形状的 PVSA TensorRT 插件冒烟引擎，"
         "用于验证插件注册、形状推导和引擎序列化。\n\n"
      << "选项：\n"
      << "  --output PATH             引擎输出路径，默认 pvsa_plugin_smoke.engine\n"
      << "  --batch N                 批大小，默认 1\n"
      << "  --num-heads N             头数，默认 8\n"
      << "  --qk-dim N                Q/K 通道数，默认 256\n"
      << "  --dim N                   V/输出通道数，默认 256\n"
      << "  --height N                特征图高度，默认 56\n"
      << "  --width N                 特征图宽度，默认 56\n"
      << "  --kv-len N                每个窗口 KV token 数，默认 64\n"
      << "  --topk N                  路由候选数，默认 8\n"
      << "  --route-p FLOAT           路由累计概率阈值，默认 0.9\n"
      << "  --route-temperature FLOAT 路由温度，默认 1.0\n"
      << "  --route-energy FLOAT      路由能量系数，默认 1.0\n"
      << "  --route-scale FLOAT       路由缩放系数，默认 1.0\n"
      << "  --flash-scale FLOAT       注意力缩放系数，默认 1.0\n"
      << "  --use-route-weight 0|1    是否使用路由权重，默认 1\n"
      << "  --help                    显示帮助\n";
}

bool take_value(int argc, char** argv, int* index, const char* option,
                std::string* value) {
  if (*index + 1 >= argc || std::string(argv[*index]) != option) {
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

template <typename T>
bool parse_number(const std::string& text, T* value) {
  try {
    if constexpr (std::is_same<T, int32_t>::value) {
      size_t consumed = 0;
      const long parsed = std::stol(text, &consumed);
      if (consumed != text.size()) {
        return false;
      }
      *value = static_cast<int32_t>(parsed);
    } else {
      size_t consumed = 0;
      const float parsed = std::stof(text, &consumed);
      if (consumed != text.size()) {
        return false;
      }
      *value = static_cast<T>(parsed);
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool parse_options(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      return false;
    }
    std::string value;
#define PARSE_INT_OPTION(name, member)                                             \
    if (take_value(argc, argv, &i, name, &value)) {                                \
      if (!parse_number<int32_t>(value, &options->member)) return false;           \
      continue;                                                                    \
    }
#define PARSE_FLOAT_OPTION(name, member)                                           \
    if (take_value(argc, argv, &i, name, &value)) {                                \
      if (!parse_number<float>(value, &options->member)) return false;             \
      continue;                                                                    \
    }
    if (take_value(argc, argv, &i, "--output", &options->output)) continue;
    PARSE_INT_OPTION("--batch", batch)
    PARSE_INT_OPTION("--num-heads", num_heads)
    PARSE_INT_OPTION("--qk-dim", qk_dim)
    PARSE_INT_OPTION("--dim", dim)
    PARSE_INT_OPTION("--n-win", n_win)
    PARSE_INT_OPTION("--height", height)
    PARSE_INT_OPTION("--width", width)
    PARSE_INT_OPTION("--kv-len", kv_len)
    PARSE_INT_OPTION("--topk", topk)
    PARSE_FLOAT_OPTION("--route-p", route_p)
    PARSE_FLOAT_OPTION("--route-temperature", route_temperature)
    PARSE_FLOAT_OPTION("--route-energy", route_energy)
    PARSE_FLOAT_OPTION("--route-scale", route_scale)
    PARSE_FLOAT_OPTION("--flash-scale", flash_scale)
    if (take_value(argc, argv, &i, "--use-route-weight", &value)) {
      int32_t flag = 0;
      if (!parse_number<int32_t>(value, &flag) || (flag != 0 && flag != 1)) {
        return false;
      }
      options->use_route_weight = flag != 0;
      continue;
    }
    std::cerr << "未知选项或缺少参数：" << argument << '\n';
    return false;
#undef PARSE_INT_OPTION
#undef PARSE_FLOAT_OPTION
  }
  return true;
}

nvinfer1::Dims make_dims(std::initializer_list<int32_t> values) {
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(values.size());
  int index = 0;
  for (const int32_t value : values) {
    dims.d[index++] = value;
  }
  return dims;
}

nvinfer1::IPluginCreator* find_plugin_creator(
    nvinfer1::IBuilder* builder, const char* name) {
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  auto* registry = getPluginRegistry();
  return registry == nullptr ? nullptr
                             : registry->getPluginCreator(name, "1", "");
#else
  if (builder == nullptr) {
    return nullptr;
  }
  auto* creator_interface =
      builder->getPluginRegistry().getCreator(name, "1", "");
  return dynamic_cast<nvinfer1::IPluginCreator*>(creator_interface);
#endif
}

nvinfer1::IPluginV2* create_route_plugin(nvinfer1::IPluginCreator* creator,
                                         const Options& options) {
  int32_t full_route = 0;
  nvinfer1::PluginField fields[] = {
      {"topk", &options.topk, nvinfer1::PluginFieldType::kINT32, 1},
      {"p", &options.route_p, nvinfer1::PluginFieldType::kFLOAT32, 1},
      {"temperature", &options.route_temperature,
       nvinfer1::PluginFieldType::kFLOAT32, 1},
      {"energy", &options.route_energy, nvinfer1::PluginFieldType::kFLOAT32,
       1},
      {"scale", &options.route_scale, nvinfer1::PluginFieldType::kFLOAT32, 1},
      {"full_route", &full_route, nvinfer1::PluginFieldType::kINT32, 1},
  };
  nvinfer1::PluginFieldCollection collection{
      static_cast<int>(sizeof(fields) / sizeof(fields[0])), fields};
  return creator == nullptr
             ? nullptr
             : creator->createPlugin("pvsa_route", &collection);
}

nvinfer1::IPluginV2* create_flash_plugin(nvinfer1::IPluginCreator* creator,
                                         const Options& options) {
  const int32_t use_route_weight = options.use_route_weight ? 1 : 0;
  nvinfer1::PluginField fields[] = {
      {"num_heads", &options.num_heads, nvinfer1::PluginFieldType::kINT32, 1},
      {"qk_dim", &options.qk_dim, nvinfer1::PluginFieldType::kINT32, 1},
      {"dim", &options.dim, nvinfer1::PluginFieldType::kINT32, 1},
      {"n_win", &options.n_win, nvinfer1::PluginFieldType::kINT32, 1},
      {"height", &options.height, nvinfer1::PluginFieldType::kINT32, 1},
      {"width", &options.width, nvinfer1::PluginFieldType::kINT32, 1},
      {"scale", &options.flash_scale, nvinfer1::PluginFieldType::kFLOAT32, 1},
      {"use_route_weight", &use_route_weight,
       nvinfer1::PluginFieldType::kINT32, 1},
  };
  nvinfer1::PluginFieldCollection collection{
      static_cast<int>(sizeof(fields) / sizeof(fields[0])), fields};
  return creator == nullptr
             ? nullptr
             : creator->createPlugin("pvsa_flash", &collection);
}

bool check_options(const Options& options) {
  if (options.batch <= 0 || options.n_win != 7 || options.height <= 0 ||
      options.width <= 0 || options.height % options.n_win != 0 ||
      options.width % options.n_win != 0 || options.kv_len <= 0 ||
      options.topk < 1 || options.topk > 49 || options.qk_dim != options.dim ||
      options.num_heads <= 0 || options.qk_dim % options.num_heads != 0 ||
      options.qk_dim / options.num_heads != 32 ||
      (options.qk_dim != 64 && options.qk_dim != 128 &&
       options.qk_dim != 256 && options.qk_dim != 512) ||
      (options.num_heads != 2 && options.num_heads != 4 &&
       options.num_heads != 8 && options.num_heads != 16)) {
    std::cerr << "参数不满足当前固定 CUDA kernel 的约束。\n";
    return false;
  }
  return true;
}

template <typename T>
void release_builder_object(T*& object) {
  if (object == nullptr) {
    return;
  }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  delete object;
#else
  object->destroy();
#endif
  object = nullptr;
}

bool write_engine(const std::string& path, nvinfer1::IHostMemory* serialized) {
  if (serialized == nullptr) {
    return false;
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    std::cerr << "无法写入引擎文件：" << path << '\n';
    return false;
  }
  output.write(static_cast<const char*>(serialized->data()),
               static_cast<std::streamsize>(serialized->size()));
  return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    return argc > 1 && (std::string(argv[1]) == "--help" ||
                         std::string(argv[1]) == "-h")
               ? 0
               : 2;
  }
  if (!check_options(options)) {
    print_usage(argv[0]);
    return 2;
  }

  const int32_t q_len =
      (options.height / options.n_win) * (options.width / options.n_win);
  Logger logger;
  auto* builder = nvinfer1::createInferBuilder(logger);
  if (builder == nullptr) {
    std::cerr << "创建 TensorRT 构建器失败。\n";
    return 1;
  }
  uint32_t network_flags = 0U;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  network_flags = 1U << static_cast<uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  auto* network = builder->createNetworkV2(network_flags);
  auto* config = builder->createBuilderConfig();
  if (network == nullptr || config == nullptr) {
    std::cerr << "创建 TensorRT 网络或构建配置失败。\n";
    release_builder_object(config);
    release_builder_object(network);
    release_builder_object(builder);
    return 1;
  }

#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR) && \
    (NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4))
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             1ULL << 30);
#else
  config->setMaxWorkspaceSize(1ULL << 30);
#endif

  auto* query = network->addInput(
      "query", nvinfer1::DataType::kFLOAT,
      make_dims({options.batch, 49, options.qk_dim}));
  auto* key = network->addInput(
      "key", nvinfer1::DataType::kFLOAT,
      make_dims({options.batch, 49, options.qk_dim}));
  auto* q_pix = network->addInput(
      "q_pix", nvinfer1::DataType::kFLOAT,
      make_dims({options.batch, 49, q_len, options.qk_dim}));
  auto* kv_pix = network->addInput(
      "kv_pix", nvinfer1::DataType::kFLOAT,
      make_dims({options.batch, 49, options.kv_len,
                 options.qk_dim + options.dim}));
  if (query == nullptr || key == nullptr || q_pix == nullptr || kv_pix == nullptr) {
    std::cerr << "创建网络输入失败。\n";
    release_builder_object(config);
    release_builder_object(network);
    release_builder_object(builder);
    return 1;
  }

  auto* route_creator = find_plugin_creator(builder, "PVSA_TopP_Route");
  auto* flash_creator = find_plugin_creator(builder, "PVSA_TopP_Flash");
  auto* route_plugin = create_route_plugin(route_creator, options);
  auto* flash_plugin = create_flash_plugin(flash_creator, options);
  if (route_plugin == nullptr || flash_plugin == nullptr) {
    std::cerr << "找不到 PVSA 插件创建器。请确认已加载"
                 " libpvsa_tensorrt_plugins.so。\n";
    if (route_plugin != nullptr) route_plugin->destroy();
    if (flash_plugin != nullptr) flash_plugin->destroy();
    release_builder_object(config);
    release_builder_object(network);
    release_builder_object(builder);
    return 1;
  }

  nvinfer1::ITensor* route_inputs[] = {query, key};
  auto* route_layer = network->addPluginV2(route_inputs, 2, *route_plugin);
  if (route_layer == nullptr) {
    std::cerr << "添加 PVSA_TopP_Route 插件层失败。\n";
    release_builder_object(config);
    release_builder_object(network);
    release_builder_object(builder);
    return 1;
  }
  route_layer->getOutput(0)->setName("route_weight");
  route_layer->getOutput(1)->setName("route_idx");
  route_layer->getOutput(2)->setName("keep_len");

  nvinfer1::ITensor* flash_inputs[] = {
      q_pix, kv_pix, route_layer->getOutput(0), route_layer->getOutput(1),
      route_layer->getOutput(2)};
  auto* flash_layer = network->addPluginV2(flash_inputs, 5, *flash_plugin);
  if (flash_layer == nullptr) {
    std::cerr << "添加 PVSA_TopP_Flash 插件层失败。\n";
    release_builder_object(config);
    release_builder_object(network);
    release_builder_object(builder);
    return 1;
  }
  flash_layer->getOutput(0)->setName("attention_output");
  network->markOutput(*flash_layer->getOutput(0));

  nvinfer1::IHostMemory* serialized =
      builder->buildSerializedNetwork(*network, *config);
  const bool success = write_engine(options.output, serialized);
  release_builder_object(serialized);
  release_builder_object(config);
  release_builder_object(network);
  release_builder_object(builder);
  if (!success) {
    std::cerr << "构建 TensorRT 引擎失败。\n";
    return 1;
  }
  std::cout << "引擎已写入：" << options.output << '\n'
            << "固定输入：batch=" << options.batch << ", H=" << options.height
            << ", W=" << options.width << ", q_len=" << q_len
            << ", kv_len=" << options.kv_len << '\n';
  return 0;
}
