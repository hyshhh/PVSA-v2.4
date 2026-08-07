#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace pvsa_tensorrt {

constexpr int kWindowSize = 7;
constexpr int kWindowsPerImage = kWindowSize * kWindowSize;
constexpr int kMaxTopK = 49;
constexpr int kHeadDim = 32;

bool can_use_specialized_kernel(int64_t p2,
                                int64_t topk,
                                int64_t num_heads,
                                int64_t qk_dim,
                                int64_t dim,
                                int64_t n_win,
                                int64_t height,
                                int64_t width);

cudaError_t launch_route(const float* query,
                         const float* key,
                         float* route_weight,
                         int32_t* route_idx,
                         int32_t* route_keep_len,
                         int64_t n,
                         int64_t qk_dim,
                         int64_t topk,
                         float p,
                         float temperature,
                         float energy,
                         float scale,
                         bool full_route,
                         cudaStream_t stream);

cudaError_t launch_flash(const float* q_pix,
                         const float* kv_pix,
                         const float* r_weight,
                         const int32_t* r_idx,
                         const int32_t* keep_len,
                         float* out,
                         int64_t n,
                         int64_t q_len,
                         int64_t kv_len,
                         int64_t topk,
                         int64_t num_heads,
                         int64_t qk_dim,
                         int64_t dim,
                         float scale,
                         int64_t n_win,
                         int64_t height,
                         int64_t width,
                         bool use_route_weight,
                         cudaStream_t stream);

}  // namespace pvsa_tensorrt
