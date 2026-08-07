#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace pvsa_tensorrt {

class TopPRoutePlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
  TopPRoutePlugin(int32_t topk, float p, float temperature, float energy,
                  float scale, bool full_route);
  TopPRoutePlugin(const void* serialData, size_t serialLength);

  int getNbOutputs() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  nvinfer1::Dims getOutputDimensions(
      int outputIndex, const nvinfer1::Dims* inputs,
      int nbInputs) noexcept override;
#endif
  bool supportsFormatCombination(
      int pos, const nvinfer1::PluginTensorDesc* inOut,
      int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  void configureWithFormat(const nvinfer1::PluginTensorDesc* in, int nbInputs,
                           const nvinfer1::PluginTensorDesc* out,
                           int nbOutputs, nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) noexcept override;
#endif
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs,
              void* workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                     const bool* inputIsBroadcasted,
                                     int nbInputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
#endif
  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* allocator) noexcept override;
  void detachFromContext() noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

private:
  int32_t topk_;
  float p_;
  float temperature_;
  float energy_;
  float scale_;
  bool full_route_;
  std::string namespace_;
};

class TopPFlashPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
  TopPFlashPlugin(int32_t numHeads, int32_t qkDim, int32_t dim,
                  int32_t nWin, int32_t height, int32_t width, float scale,
                  bool useRouteWeight);
  TopPFlashPlugin(const void* serialData, size_t serialLength);

  int getNbOutputs() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  nvinfer1::Dims getOutputDimensions(
      int outputIndex, const nvinfer1::Dims* inputs,
      int nbInputs) noexcept override;
#endif
  bool supportsFormatCombination(
      int pos, const nvinfer1::PluginTensorDesc* inOut,
      int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  void configureWithFormat(const nvinfer1::PluginTensorDesc* in, int nbInputs,
                           const nvinfer1::PluginTensorDesc* out,
                           int nbOutputs, nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) noexcept override;
#endif
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs,
              void* workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const noexcept override;
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                     const bool* inputIsBroadcasted,
                                     int nbInputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
#endif
  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* allocator) noexcept override;
  void detachFromContext() noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

private:
  int32_t num_heads_;
  int32_t qk_dim_;
  int32_t dim_;
  int32_t n_win_;
  int32_t height_;
  int32_t width_;
  float scale_;
  bool use_route_weight_;
  std::string namespace_;
};

}  // namespace pvsa_tensorrt
