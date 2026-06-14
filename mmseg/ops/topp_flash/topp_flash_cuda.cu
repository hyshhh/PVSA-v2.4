#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;

// ============================================================================
// 类型转换
// ============================================================================
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// ============================================================================
// Warp reduce sum（无同步，最快）
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// ============================================================================
// 优化 kernel v2：每个线程处理一个 (coarse, head, q_pos)，独立计算完整 Q*K
// 优势：无 warp shuffle 开销，适合 head_q <= 32 的场景
// ============================================================================
template <typename scalar_t>
__global__ void topp_flash_kernel(
    const scalar_t *__restrict__ q_pix,
    const scalar_t *__restrict__ kv_pix,
    const scalar_t *__restrict__ r_weight,
    const int64_t *__restrict__ r_idx,
    const int64_t *__restrict__ keep_len,
    float *__restrict__ out,
    int64_t n,
    int64_t p2,
    int64_t q_len,
    int64_t kv_len,
    int64_t topk,
    int64_t num_heads,
    int64_t qk_dim,
    int64_t dim,
    float scale,
    int64_t coarse_total) {
  
  // Grid: (coarse_total * num_heads * q_len,)
  // Block: (256,)
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_tasks = coarse_total * num_heads * q_len;
  
  if (tid >= total_tasks) return;
  
  // 解析 (coarse, head, q_pos)
  const int64_t coarse = tid / (num_heads * q_len);
  const int rem = tid % (num_heads * q_len);
  const int64_t head = rem / q_len;
  const int64_t q_pos = rem % q_len;
  
  const int64_t batch = coarse / p2;
  const int64_t p = coarse % p2;
  const int64_t head_v = dim / num_heads;
  const int64_t head_q = qk_dim / num_heads;
  const int64_t route_base = coarse * topk;
  
  // 加载 Q 到寄存器
  const int64_t q_base = ((batch * p2 + p) * q_len + q_pos) * qk_dim + head * head_q;
  float q_local[32];  // 最多支持 head_q=32
  for (int d = 0; d < head_q && d < 32; d++) {
    q_local[d] = to_float(q_pix[q_base + d]);
  }
  
  // 有效 topk
  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;
  
  if (valid_topk <= 0) {
    for (int v_ch = 0; v_ch < head_v; v_ch++) {
      int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim + head * head_v + v_ch;
      out[out_idx] = 0.0f;
    }
    return;
  }
  
  // 在线 softmax 状态
  float o_acc[32] = {0.0f};  // 最多支持 head_v=32
  float m_prev = -INFINITY;
  float l_prev = 0.0f;
  
  // 遍历所有 KV 位置
  for (int64_t tk = 0; tk < valid_topk; tk++) {
    const int64_t kv_window = r_idx[route_base + tk];
    const float route_weight = to_float(r_weight[route_base + tk]);
    
    for (int64_t kv_pos = 0; kv_pos < kv_len; kv_pos++) {
      const int64_t kv_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
      
      // 独立计算 Q*K（无 warp shuffle）
      float score = 0.0f;
      for (int d = 0; d < head_q && d < 32; d++) {
        float k_val = to_float(kv_pix[kv_base + head * head_q + d]) * route_weight;
        score += q_local[d] * k_val;
      }
      score *= scale;
      
      // 在线 softmax
      float m_new = fmaxf(m_prev, score);
      float exp_prev = expf(m_prev - m_new);
      float exp_new = expf(score - m_new);
      l_prev = l_prev * exp_prev + exp_new;
      
      // 更新 V 通道
      for (int v_ch = 0; v_ch < head_v && v_ch < 32; v_ch++) {
        float v_val = to_float(kv_pix[kv_base + qk_dim + head * head_v + v_ch]) * route_weight;
        o_acc[v_ch] = o_acc[v_ch] * exp_prev + exp_new * v_val;
      }
      
      m_prev = m_new;
    }
  }
  
  // 输出
  for (int v_ch = 0; v_ch < head_v && v_ch < 32; v_ch++) {
    int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim + head * head_v + v_ch;
    out[out_idx] = o_acc[v_ch] / fmaxf(l_prev, 1e-20f);
  }
}

// ============================================================================
// unflatten kernel
// ============================================================================
__global__ void unflatten_windows_kernel(const float *__restrict__ flat,
                                         float *__restrict__ out,
                                         int64_t n,
                                         int64_t n_win,
                                         int64_t height,
                                         int64_t width,
                                         int64_t dim,
                                         int64_t total) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) return;

  int64_t c = linear % dim;
  int64_t w = (linear / dim) % width;
  int64_t h = (linear / (dim * width)) % height;
  int64_t batch = linear / (dim * width * height);

  int64_t q_h = height / n_win;
  int64_t q_w = width / n_win;
  int64_t win_y = h / q_h;
  int64_t win_x = w / q_w;
  int64_t local_y = h % q_h;
  int64_t local_x = w % q_w;
  int64_t p = win_y * n_win + win_x;
  int64_t q_pos = local_y * q_w + local_x;

  int64_t flat_index = ((batch * n_win * n_win + p) * (q_h * q_w) + q_pos) * dim + c;
  out[linear] = flat[flat_index];
}

// ============================================================================
// 入口函数
// ============================================================================
torch::Tensor topp_flash_forward_cuda(torch::Tensor q_pix,
                                      torch::Tensor kv_pix,
                                      torch::Tensor r_weight,
                                      torch::Tensor r_idx,
                                      torch::Tensor keep_len,
                                      int64_t num_heads,
                                      int64_t qk_dim,
                                      int64_t dim,
                                      double scale,
                                      int64_t n_win,
                                      int64_t height,
                                      int64_t width) {
  const auto n = q_pix.size(0);
  const auto p2 = q_pix.size(1);
  const auto q_len = q_pix.size(2);
  const auto kv_len = kv_pix.size(2);
  const auto topk = r_idx.size(2);

  auto flat_out = torch::empty({n, p2, q_len, dim}, q_pix.options().dtype(torch::kFloat32));
  auto out = torch::empty({n, height, width, dim}, q_pix.options().dtype(torch::kFloat32));

  const int64_t coarse_total = n * p2;
  const int64_t total_tasks = coarse_total * num_heads * q_len;
  const int threads = 256;
  const int blocks = (total_tasks + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (q_pix.scalar_type() == torch::kFloat16) {
    topp_flash_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(q_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kv_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(r_weight.data_ptr<at::Half>()),
        r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), coarse_total);
  } else if (q_pix.scalar_type() == torch::kBFloat16) {
    topp_flash_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kv_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(r_weight.data_ptr<at::BFloat16>()),
        r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), coarse_total);
  } else {
    topp_flash_kernel<float><<<blocks, threads, 0, stream>>>(
        q_pix.data_ptr<float>(), kv_pix.data_ptr<float>(),
        r_weight.data_ptr<float>(),
        r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), coarse_total);
  }

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + 255) / 256);
  unflatten_windows_kernel<<<out_blocks, 256, 0, stream>>>(
      flat_out.data_ptr<float>(), out.data_ptr<float>(),
      n, n_win, height, width, dim, out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
