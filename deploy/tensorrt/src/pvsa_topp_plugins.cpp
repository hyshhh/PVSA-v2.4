#include "pvsa_topp_plugins.h"
#include "pvsa_topp_kernel.cuh"

#include <NvInferPlugin.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

constexpr int kRouteInputCount = 2;
constexpr int kRouteOutputCount = 3;
constexpr int kFlashInputCount = 5;
constexpr int kFlashOutputCount = 1;
constexpr int kP2 = pvsa_tensorrt::kWindowsPerImage;
constexpr int kMaxTopK = pvsa_tensorrt::kMaxTopK;

int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

bool is_supported_qk_dim(int64_t qk_dim) noexcept {
  return qk_dim == 64 || qk_dim == 128 || qk_dim == 256 || qk_dim == 512;
}

bool is_supported_num_heads(int64_t num_heads) noexcept {
  return num_heads == 2 || num_heads == 4 || num_heads == 8 ||
         num_heads == 16;
}

bool is_finite(float value) noexcept {
  return std::isfinite(static_cast<double>(value));
}

bool valid_route_config(int32_t topk, float p, float temperature,
                        float energy, float scale, bool full_route) noexcept {
  if (topk < 1 || topk > kMaxTopK) {
    return false;
  }
  // 与原始 PyTorch CUDA 路径保持一致：full_route 只允许输出完整 49 路。
  if (full_route && topk != kMaxTopK) {
    return false;
  }
  return p >= 0.0f && p <= 1.0f && temperature > 0.0f &&
         is_finite(p) && is_finite(temperature) && is_finite(energy) &&
         is_finite(scale);
}

bool valid_flash_config(int32_t num_heads, int32_t qk_dim, int32_t dim,
                        int32_t n_win, int32_t height, int32_t width) noexcept {
  if (n_win != pvsa_tensorrt::kWindowSize || height <= 0 || width <= 0 ||
      height % pvsa_tensorrt::kWindowSize != 0 ||
      width % pvsa_tensorrt::kWindowSize != 0) {
    return false;
  }
  if (!is_supported_num_heads(num_heads) || qk_dim != dim ||
      !is_supported_qk_dim(qk_dim)) {
    return false;
  }
  return qk_dim % num_heads == 0 && dim % num_heads == 0 &&
         qk_dim / num_heads == pvsa_tensorrt::kHeadDim &&
         dim / num_heads == pvsa_tensorrt::kHeadDim;
}

bool is_linear_float(const nvinfer1::PluginTensorDesc& desc) noexcept {
  return desc.type == nvinfer1::DataType::kFLOAT &&
         desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool is_linear_int32(const nvinfer1::PluginTensorDesc& desc) noexcept {
  return desc.type == nvinfer1::DataType::kINT32 &&
         desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool same_dim(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) noexcept {
  if (lhs.nbDims != rhs.nbDims) {
    return false;
  }
  for (int i = 0; i < lhs.nbDims; ++i) {
    if (lhs.d[i] != rhs.d[i]) {
      return false;
    }
  }
  return true;
}

nvinfer1::Dims route_output_dims(int output_index,
                                 const nvinfer1::Dims* inputs,
                                 int32_t topk) noexcept {
  nvinfer1::Dims output{};
  if (inputs == nullptr || output_index < 0 || output_index >= kRouteOutputCount ||
      inputs[0].nbDims != 3) {
    return output;
  }
  if (output_index == 2) {
    output.nbDims = 2;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
  } else {
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = topk;
  }
  return output;
}

template <typename T>
void write_value(char*& buffer, const T& value) noexcept {
  std::memcpy(buffer, &value, sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
bool read_value(const char*& buffer, size_t& remaining, T* value) noexcept {
  if (value == nullptr || remaining < sizeof(T)) {
    return false;
  }
  std::memcpy(value, buffer, sizeof(T));
  buffer += sizeof(T);
  remaining -= sizeof(T);
  return true;
}

bool read_field_int32(const nvinfer1::PluginField& field, int32_t* value) {
  if (value == nullptr || field.data == nullptr || field.length < 1 ||
      field.type != nvinfer1::PluginFieldType::kINT32) {
    return false;
  }
  *value = *static_cast<const int32_t*>(field.data);
  return true;
}

bool read_field_float(const nvinfer1::PluginField& field, float* value) {
  if (value == nullptr || field.data == nullptr || field.length < 1 ||
      field.type != nvinfer1::PluginFieldType::kFLOAT32) {
    return false;
  }
  *value = *static_cast<const float*>(field.data);
  return true;
}

}  // namespace

namespace pvsa_tensorrt {

TopPRoutePlugin::TopPRoutePlugin(int32_t topk, float p, float temperature,
                                 float energy, float scale, bool full_route)
    : topk_(topk),
      p_(p),
      temperature_(temperature),
      energy_(energy),
      scale_(scale),
      full_route_(full_route) {}

TopPRoutePlugin::TopPRoutePlugin(const void* serial_data, size_t serial_length)
    : topk_(0), p_(0.0f), temperature_(1.0f), energy_(1.0f), scale_(1.0f),
      full_route_(false) {
  const char* buffer = static_cast<const char*>(serial_data);
  size_t remaining = serial_length;
  uint8_t full_route = 0;
  if (buffer == nullptr ||
      !read_value(buffer, remaining, &topk_) ||
      !read_value(buffer, remaining, &p_) ||
      !read_value(buffer, remaining, &temperature_) ||
      !read_value(buffer, remaining, &energy_) ||
      !read_value(buffer, remaining, &scale_) ||
      !read_value(buffer, remaining, &full_route)) {
    topk_ = 0;
    return;
  }
  full_route_ = full_route != 0;
}

int TopPRoutePlugin::getNbOutputs() const noexcept { return kRouteOutputCount; }

nvinfer1::DimsExprs TopPRoutePlugin::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  nvinfer1::DimsExprs output{};
  if (inputs == nullptr || nb_inputs != kRouteInputCount ||
      output_index < 0 || output_index >= kRouteOutputCount ||
      inputs[0].nbDims != 3 || inputs[1].nbDims != 3) {
    return output;
  }
  output.nbDims = output_index == 2 ? 2 : 3;
  output.d[0] = inputs[0].d[0];
  output.d[1] = inputs[0].d[1];
  if (output_index != 2) {
    output.d[2] = expr_builder.constant(topk_);
  }
  return output;
}

nvinfer1::Dims TopPRoutePlugin::getOutputDimensions(
    int output_index, const nvinfer1::Dims* inputs, int nb_inputs) noexcept {
  if (inputs == nullptr || nb_inputs != kRouteInputCount) {
    return nvinfer1::Dims{};
  }
  return route_output_dims(output_index, inputs, topk_);
}

bool TopPRoutePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) noexcept {
  (void)nb_inputs;
  (void)nb_outputs;
  if (in_out == nullptr || pos < 0 || pos >= kRouteInputCount + kRouteOutputCount) {
    return false;
  }
  if (pos <= 2) {
    return is_linear_float(in_out[pos]);
  }
  return is_linear_int32(in_out[pos]);
}

void TopPRoutePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
}

void TopPRoutePlugin::configureWithFormat(
    const nvinfer1::PluginTensorDesc* in, int nb_inputs,
    const nvinfer1::PluginTensorDesc* out, int nb_outputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format,
    int max_batch_size) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
  (void)type;
  (void)format;
  (void)max_batch_size;
}

int TopPRoutePlugin::initialize() noexcept { return 0; }
void TopPRoutePlugin::terminate() noexcept {}

size_t TopPRoutePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const noexcept {
  (void)inputs;
  (void)nb_inputs;
  (void)outputs;
  (void)nb_outputs;
  return 0;
}

int TopPRoutePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  (void)workspace;
  if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr || outputs[1] == nullptr || outputs[2] == nullptr ||
      input_desc[0].dims.nbDims != 3 || input_desc[1].dims.nbDims != 3 ||
      !is_linear_float(input_desc[0]) || !is_linear_float(input_desc[1]) ||
      !is_linear_float(output_desc[0]) || !is_linear_int32(output_desc[1]) ||
      !is_linear_int32(output_desc[2])) {
    return invalid_value();
  }

  const nvinfer1::Dims& query_dims = input_desc[0].dims;
  const nvinfer1::Dims& key_dims = input_desc[1].dims;
  if (!same_dim(query_dims, key_dims) || query_dims.d[0] <= 0 ||
      query_dims.d[1] != kP2 || !is_supported_qk_dim(query_dims.d[2]) ||
      !valid_route_config(topk_, p_, temperature_, energy_, scale_,
                          full_route_)) {
    return invalid_value();
  }
  const int64_t n = query_dims.d[0];
  const int64_t qk_dim = query_dims.d[2];
  const nvinfer1::Dims expected_route = route_output_dims(0, &query_dims, topk_);
  const nvinfer1::Dims expected_index = route_output_dims(1, &query_dims, topk_);
  const nvinfer1::Dims expected_keep = route_output_dims(2, &query_dims, topk_);
  if (!same_dim(output_desc[0].dims, expected_route) ||
      !same_dim(output_desc[1].dims, expected_index) ||
      !same_dim(output_desc[2].dims, expected_keep)) {
    return invalid_value();
  }

  const cudaError_t error = launch_route(
      static_cast<const float*>(inputs[0]),
      static_cast<const float*>(inputs[1]), static_cast<float*>(outputs[0]),
      static_cast<int32_t*>(outputs[1]), static_cast<int32_t*>(outputs[2]), n,
      qk_dim, topk_, p_, temperature_, energy_, scale_, full_route_, stream);
  return static_cast<int>(error);
}

size_t TopPRoutePlugin::getSerializationSize() const noexcept {
  return sizeof(topk_) + sizeof(p_) + sizeof(temperature_) + sizeof(energy_) +
         sizeof(scale_) + sizeof(uint8_t);
}

void TopPRoutePlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  char* cursor = static_cast<char*>(buffer);
  write_value(cursor, topk_);
  write_value(cursor, p_);
  write_value(cursor, temperature_);
  write_value(cursor, energy_);
  write_value(cursor, scale_);
  const uint8_t full_route = full_route_ ? 1 : 0;
  write_value(cursor, full_route);
}

nvinfer1::IPluginV2DynamicExt* TopPRoutePlugin::clone() const noexcept {
  auto* plugin = new TopPRoutePlugin(topk_, p_, temperature_, energy_, scale_,
                                     full_route_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DataType TopPRoutePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  (void)input_types;
  (void)nb_inputs;
  return index == 0 ? nvinfer1::DataType::kFLOAT
                    : nvinfer1::DataType::kINT32;
}

bool TopPRoutePlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool* input_is_broadcasted,
    int nb_inputs) const noexcept {
  (void)output_index;
  (void)input_is_broadcasted;
  (void)nb_inputs;
  return false;
}

bool TopPRoutePlugin::canBroadcastInputAcrossBatch(int input_index) const noexcept {
  (void)input_index;
  return false;
}

void TopPRoutePlugin::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas,
    nvinfer1::IGpuAllocator* allocator) noexcept {
  (void)cudnn;
  (void)cublas;
  (void)allocator;
}

void TopPRoutePlugin::detachFromContext() noexcept {}
const char* TopPRoutePlugin::getPluginType() const noexcept {
  return "PVSA_TopP_Route";
}
const char* TopPRoutePlugin::getPluginVersion() const noexcept { return "1"; }
void TopPRoutePlugin::destroy() noexcept { delete this; }
void TopPRoutePlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}
const char* TopPRoutePlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

TopPFlashPlugin::TopPFlashPlugin(int32_t num_heads, int32_t qk_dim,
                                 int32_t dim, int32_t n_win, int32_t height,
                                 int32_t width, float scale,
                                 bool use_route_weight)
    : num_heads_(num_heads),
      qk_dim_(qk_dim),
      dim_(dim),
      n_win_(n_win),
      height_(height),
      width_(width),
      scale_(scale),
      use_route_weight_(use_route_weight) {}

TopPFlashPlugin::TopPFlashPlugin(const void* serial_data, size_t serial_length)
    : num_heads_(0),
      qk_dim_(0),
      dim_(0),
      n_win_(0),
      height_(0),
      width_(0),
      scale_(0.0f),
      use_route_weight_(false) {
  const char* buffer = static_cast<const char*>(serial_data);
  size_t remaining = serial_length;
  uint8_t use_route_weight = 0;
  if (buffer == nullptr ||
      !read_value(buffer, remaining, &num_heads_) ||
      !read_value(buffer, remaining, &qk_dim_) ||
      !read_value(buffer, remaining, &dim_) ||
      !read_value(buffer, remaining, &n_win_) ||
      !read_value(buffer, remaining, &height_) ||
      !read_value(buffer, remaining, &width_) ||
      !read_value(buffer, remaining, &scale_) ||
      !read_value(buffer, remaining, &use_route_weight)) {
    num_heads_ = 0;
    return;
  }
  use_route_weight_ = use_route_weight != 0;
}

int TopPFlashPlugin::getNbOutputs() const noexcept { return kFlashOutputCount; }

nvinfer1::DimsExprs TopPFlashPlugin::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  (void)expr_builder;
  nvinfer1::DimsExprs output{};
  if (inputs == nullptr || nb_inputs != kFlashInputCount || output_index != 0 ||
      inputs[0].nbDims != 4 || inputs[1].nbDims != 4 ||
      inputs[2].nbDims != 3 || inputs[3].nbDims != 3 ||
      inputs[4].nbDims != 2) {
    return output;
  }
  output.nbDims = 4;
  output.d[0] = inputs[0].d[0];
  output.d[1] = expr_builder.constant(height_);
  output.d[2] = expr_builder.constant(width_);
  output.d[3] = expr_builder.constant(dim_);
  return output;
}

nvinfer1::Dims TopPFlashPlugin::getOutputDimensions(
    int output_index, const nvinfer1::Dims* inputs, int nb_inputs) noexcept {
  if (inputs == nullptr || nb_inputs != kFlashInputCount || output_index != 0) {
    return nvinfer1::Dims{};
  }
  nvinfer1::Dims output{};
  output.nbDims = 4;
  output.d[0] = inputs[0].d[0];
  output.d[1] = height_;
  output.d[2] = width_;
  output.d[3] = dim_;
  return output;
}

bool TopPFlashPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) noexcept {
  (void)nb_inputs;
  (void)nb_outputs;
  if (in_out == nullptr || pos < 0 || pos >= kFlashInputCount + kFlashOutputCount) {
    return false;
  }
  if (pos <= 2 || pos == 5) {
    return is_linear_float(in_out[pos]);
  }
  return is_linear_int32(in_out[pos]);
}

void TopPFlashPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
}

void TopPFlashPlugin::configureWithFormat(
    const nvinfer1::PluginTensorDesc* in, int nb_inputs,
    const nvinfer1::PluginTensorDesc* out, int nb_outputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format,
    int max_batch_size) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
  (void)type;
  (void)format;
  (void)max_batch_size;
}

int TopPFlashPlugin::initialize() noexcept { return 0; }
void TopPFlashPlugin::terminate() noexcept {}

size_t TopPFlashPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const noexcept {
  (void)inputs;
  (void)nb_inputs;
  (void)outputs;
  (void)nb_outputs;
  return 0;
}

int TopPFlashPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  (void)workspace;
  if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
      outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
      inputs[2] == nullptr || inputs[3] == nullptr || inputs[4] == nullptr ||
      outputs[0] == nullptr || input_desc[0].dims.nbDims != 4 ||
      input_desc[1].dims.nbDims != 4 || input_desc[2].dims.nbDims != 3 ||
      input_desc[3].dims.nbDims != 3 || input_desc[4].dims.nbDims != 2 ||
      !is_linear_float(input_desc[0]) || !is_linear_float(input_desc[1]) ||
      !is_linear_float(input_desc[2]) || !is_linear_int32(input_desc[3]) ||
      !is_linear_int32(input_desc[4]) || !is_linear_float(output_desc[0])) {
    return invalid_value();
  }
  if (!valid_flash_config(num_heads_, qk_dim_, dim_, n_win_, height_, width_) ||
      !is_finite(scale_)) {
    return invalid_value();
  }

  const nvinfer1::Dims& q_dims = input_desc[0].dims;
  const nvinfer1::Dims& kv_dims = input_desc[1].dims;
  const nvinfer1::Dims& weight_dims = input_desc[2].dims;
  const nvinfer1::Dims& index_dims = input_desc[3].dims;
  const nvinfer1::Dims& keep_dims = input_desc[4].dims;
  const int64_t expected_q_len =
      (height_ / pvsa_tensorrt::kWindowSize) *
      (width_ / pvsa_tensorrt::kWindowSize);
  if (q_dims.d[0] <= 0 || q_dims.d[1] != kP2 ||
      q_dims.d[2] != expected_q_len || q_dims.d[3] != qk_dim_ ||
      kv_dims.d[0] != q_dims.d[0] ||
      kv_dims.d[1] != kP2 || kv_dims.d[2] <= 0 ||
      kv_dims.d[3] != qk_dim_ + dim_ || weight_dims.d[0] != q_dims.d[0] ||
      weight_dims.d[1] != kP2 || index_dims.d[0] != q_dims.d[0] ||
      index_dims.d[1] != kP2 || keep_dims.d[0] != q_dims.d[0] ||
      keep_dims.d[1] != kP2 || weight_dims.d[2] <= 0 ||
      weight_dims.d[2] > kMaxTopK || !same_dim(weight_dims, index_dims)) {
    return invalid_value();
  }
  nvinfer1::Dims expected_output{};
  expected_output.nbDims = 4;
  expected_output.d[0] = q_dims.d[0];
  expected_output.d[1] = height_;
  expected_output.d[2] = width_;
  expected_output.d[3] = dim_;
  if (!same_dim(output_desc[0].dims, expected_output)) {
    return invalid_value();
  }

  const int64_t n = q_dims.d[0];
  const int64_t q_len = q_dims.d[2];
  const int64_t kv_len = kv_dims.d[2];
  const int64_t topk = weight_dims.d[2];
  const cudaError_t error = launch_flash(
      static_cast<const float*>(inputs[0]),
      static_cast<const float*>(inputs[1]),
      static_cast<const float*>(inputs[2]),
      static_cast<const int32_t*>(inputs[3]),
      static_cast<const int32_t*>(inputs[4]), static_cast<float*>(outputs[0]),
      n, q_len, kv_len, topk, num_heads_, qk_dim_, dim_, scale_, n_win_,
      height_, width_, use_route_weight_, stream);
  return static_cast<int>(error);
}

size_t TopPFlashPlugin::getSerializationSize() const noexcept {
  return sizeof(num_heads_) + sizeof(qk_dim_) + sizeof(dim_) + sizeof(n_win_) +
         sizeof(height_) + sizeof(width_) + sizeof(scale_) + sizeof(uint8_t);
}

void TopPFlashPlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  char* cursor = static_cast<char*>(buffer);
  write_value(cursor, num_heads_);
  write_value(cursor, qk_dim_);
  write_value(cursor, dim_);
  write_value(cursor, n_win_);
  write_value(cursor, height_);
  write_value(cursor, width_);
  write_value(cursor, scale_);
  const uint8_t use_route_weight = use_route_weight_ ? 1 : 0;
  write_value(cursor, use_route_weight);
}

nvinfer1::IPluginV2DynamicExt* TopPFlashPlugin::clone() const noexcept {
  auto* plugin = new TopPFlashPlugin(num_heads_, qk_dim_, dim_, n_win_, height_,
                                     width_, scale_, use_route_weight_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DataType TopPFlashPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  (void)index;
  (void)input_types;
  (void)nb_inputs;
  return nvinfer1::DataType::kFLOAT;
}

bool TopPFlashPlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool* input_is_broadcasted,
    int nb_inputs) const noexcept {
  (void)output_index;
  (void)input_is_broadcasted;
  (void)nb_inputs;
  return false;
}

bool TopPFlashPlugin::canBroadcastInputAcrossBatch(int input_index) const noexcept {
  (void)input_index;
  return false;
}

void TopPFlashPlugin::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas,
    nvinfer1::IGpuAllocator* allocator) noexcept {
  (void)cudnn;
  (void)cublas;
  (void)allocator;
}

void TopPFlashPlugin::detachFromContext() noexcept {}
const char* TopPFlashPlugin::getPluginType() const noexcept {
  return "PVSA_TopP_Flash";
}
const char* TopPFlashPlugin::getPluginVersion() const noexcept { return "1"; }
void TopPFlashPlugin::destroy() noexcept { delete this; }
void TopPFlashPlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}
const char* TopPFlashPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

class TopPRouteCreator final : public nvinfer1::IPluginCreator {
 public:
  TopPRouteCreator() {
    fields_[0] = {"topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[1] = {"p", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields_[2] = {"temperature", nullptr,
                 nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields_[3] = {"energy", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields_[4] = {"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields_[5] = {"full_route", nullptr,
                 nvinfer1::PluginFieldType::kINT32, 1};
    collection_.nbFields = 6;
    collection_.fields = fields_;
  }

  const char* getPluginName() const noexcept override {
    return "PVSA_TopP_Route";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &collection_;
  }
  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* field_collection)
      noexcept override {
    (void)name;
    int32_t topk = 49;
    float p = 1.0f;
    float temperature = 1.0f;
    float energy = 1.0f;
    float scale = 1.0f;
    bool full_route = false;
    if (field_collection != nullptr) {
      for (int i = 0; i < field_collection->nbFields; ++i) {
        const auto& field = field_collection->fields[i];
        int32_t int_value = 0;
        float float_value = 0.0f;
        if (std::strcmp(field.name, "topk") == 0 &&
            read_field_int32(field, &int_value)) {
          topk = int_value;
        } else if (std::strcmp(field.name, "p") == 0 &&
                   read_field_float(field, &float_value)) {
          p = float_value;
        } else if (std::strcmp(field.name, "temperature") == 0 &&
                   read_field_float(field, &float_value)) {
          temperature = float_value;
        } else if (std::strcmp(field.name, "energy") == 0 &&
                   read_field_float(field, &float_value)) {
          energy = float_value;
        } else if (std::strcmp(field.name, "scale") == 0 &&
                   read_field_float(field, &float_value)) {
          scale = float_value;
        } else if (std::strcmp(field.name, "full_route") == 0 &&
                   read_field_int32(field, &int_value)) {
          full_route = int_value != 0;
        }
      }
    }
    if (!valid_route_config(topk, p, temperature, energy, scale, full_route)) {
      return nullptr;
    }
    return new TopPRoutePlugin(topk, p, temperature, energy, scale, full_route);
  }
  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data, size_t serial_length)
      noexcept override {
    (void)name;
    return new TopPRoutePlugin(serial_data, serial_length);
  }
  void setPluginNamespace(const char* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  nvinfer1::PluginField fields_[6]{};
  nvinfer1::PluginFieldCollection collection_{};
  std::string namespace_;
};

class TopPFlashCreator final : public nvinfer1::IPluginCreator {
 public:
  TopPFlashCreator() {
    fields_[0] = {"num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[1] = {"qk_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[2] = {"dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[3] = {"n_win", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[4] = {"height", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[5] = {"width", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    fields_[6] = {"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
    fields_[7] = {"use_route_weight", nullptr,
                 nvinfer1::PluginFieldType::kINT32, 1};
    collection_.nbFields = 8;
    collection_.fields = fields_;
  }

  const char* getPluginName() const noexcept override {
    return "PVSA_TopP_Flash";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &collection_;
  }
  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* field_collection)
      noexcept override {
    (void)name;
    int32_t num_heads = 8;
    int32_t qk_dim = 256;
    int32_t dim = 256;
    int32_t n_win = 7;
    int32_t height = 56;
    int32_t width = 56;
    float scale = 1.0f;
    bool use_route_weight = true;
    if (field_collection != nullptr) {
      for (int i = 0; i < field_collection->nbFields; ++i) {
        const auto& field = field_collection->fields[i];
        int32_t int_value = 0;
        float float_value = 0.0f;
        if (std::strcmp(field.name, "num_heads") == 0 &&
            read_field_int32(field, &int_value)) {
          num_heads = int_value;
        } else if (std::strcmp(field.name, "qk_dim") == 0 &&
                   read_field_int32(field, &int_value)) {
          qk_dim = int_value;
        } else if (std::strcmp(field.name, "dim") == 0 &&
                   read_field_int32(field, &int_value)) {
          dim = int_value;
        } else if (std::strcmp(field.name, "n_win") == 0 &&
                   read_field_int32(field, &int_value)) {
          n_win = int_value;
        } else if (std::strcmp(field.name, "height") == 0 &&
                   read_field_int32(field, &int_value)) {
          height = int_value;
        } else if (std::strcmp(field.name, "width") == 0 &&
                   read_field_int32(field, &int_value)) {
          width = int_value;
        } else if (std::strcmp(field.name, "scale") == 0 &&
                   read_field_float(field, &float_value)) {
          scale = float_value;
        } else if (std::strcmp(field.name, "use_route_weight") == 0 &&
                   read_field_int32(field, &int_value)) {
          use_route_weight = int_value != 0;
        }
      }
    }
    if (!valid_flash_config(num_heads, qk_dim, dim, n_win, height, width) ||
        !is_finite(scale)) {
      return nullptr;
    }
    return new TopPFlashPlugin(num_heads, qk_dim, dim, n_win, height, width,
                               scale, use_route_weight);
  }
  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data, size_t serial_length)
      noexcept override {
    (void)name;
    return new TopPFlashPlugin(serial_data, serial_length);
  }
  void setPluginNamespace(const char* plugin_namespace) noexcept override {
    namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  nvinfer1::PluginField fields_[8]{};
  nvinfer1::PluginFieldCollection collection_{};
  std::string namespace_;
};

REGISTER_TENSORRT_PLUGIN(TopPRouteCreator);
REGISTER_TENSORRT_PLUGIN(TopPFlashCreator);

}  // namespace pvsa_tensorrt
