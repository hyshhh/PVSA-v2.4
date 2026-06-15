#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int SPECIAL_N_WIN = 7;
constexpr int SPECIAL_P2 = SPECIAL_N_WIN * SPECIAL_N_WIN;
constexpr int SPECIAL_HEAD_DIM = 32;
constexpr int SPECIAL_MAX_TOPK = 49;
constexpr int SPECIAL_KV_TILE = 32;
constexpr int GENERIC_MAX_ELEMS_PER_LANE = 16;
constexpr int ROUTE_THREADS = 128;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
  return val;
}

__device__ __forceinline__ int64_t nhwc_offset(int64_t batch,
                                               int64_t h,
                                               int64_t w,
                                               int64_t c,
                                               int64_t height,
                                               int64_t width,
                                               int64_t dim) {
  return ((batch * height + h) * width + w) * dim + c;
}

__global__ void topp_route_nwin7_kernel(
    const float *__restrict__ query,
    float *__restrict__ route_weight,
    int64_t *__restrict__ route_idx,
    bool *__restrict__ route_mask,
    int *__restrict__ max_keep,
    int64_t n,
    int64_t qk_dim,
    int64_t topk,
    float p,
    float temperature,
    float energy,
    float scale) {
  __shared__ float s_scores[SPECIAL_P2];
  __shared__ float s_q_norm_acc[ROUTE_THREADS];

  const int row = blockIdx.x;
  const int batch = row / SPECIAL_P2;
  const int q_win = row - batch * SPECIAL_P2;
  const int tid = threadIdx.x;
  const int64_t q_base =
      (static_cast<int64_t>(batch) * SPECIAL_P2 + q_win) * qk_dim;

  float norm_acc = 0.0f;
  for (int64_t d = tid; d < qk_dim; d += blockDim.x) {
    const float qv = query[q_base + d];
    norm_acc += qv * qv;
  }
  s_q_norm_acc[tid] = norm_acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_q_norm_acc[tid] += s_q_norm_acc[tid + stride];
    }
    __syncthreads();
  }
  const float q_inv_norm = rsqrtf(fmaxf(s_q_norm_acc[0], 1.0e-24f));

  if (tid < SPECIAL_P2) {
    const int k_win = tid;
    const int64_t k_base =
        (static_cast<int64_t>(batch) * SPECIAL_P2 + k_win) * qk_dim;
    float dot = 0.0f;
    float k_norm = 0.0f;
    for (int64_t d = 0; d < qk_dim; d++) {
      const float qv = query[q_base + d];
      const float kv = query[k_base + d];
      dot += qv * kv;
      k_norm += kv * kv;
    }
    s_scores[k_win] =
        dot * q_inv_norm * rsqrtf(fmaxf(k_norm, 1.0e-24f)) * scale;
  }
  __syncthreads();

  if (tid == 0) {
    float selected_scores[SPECIAL_MAX_TOPK];
    int selected_idx[SPECIAL_MAX_TOPK];

    for (int tk = 0; tk < topk; tk++) {
      float best = -INFINITY;
      int best_idx = 0;
      for (int col = 0; col < SPECIAL_P2; col++) {
        const float score = s_scores[col];
        bool used = false;
        for (int prev = 0; prev < tk; prev++) {
          if (selected_idx[prev] == col) {
            used = true;
            break;
          }
        }
        if (!used && (score > best || (score == best && col < best_idx))) {
          best = score;
          best_idx = col;
        }
      }
      selected_scores[tk] = best;
      selected_idx[tk] = best_idx;
    }

    float max_score = -INFINITY;
    for (int tk = 0; tk < topk; tk++) {
      max_score = fmaxf(max_score, selected_scores[tk] / temperature);
    }
    float denom = 0.0f;
    for (int tk = 0; tk < topk; tk++) {
      selected_scores[tk] = expf(selected_scores[tk] / temperature - max_score);
      denom += selected_scores[tk];
    }

    float cumsum = 0.0f;
    int keep_len = 0;
    for (int tk = 0; tk < topk; tk++) {
      const float prob = selected_scores[tk] / denom;
      cumsum += prob;
      if (cumsum <= p) {
        keep_len++;
      }
      selected_scores[tk] = prob;
    }
    if (keep_len < 1) keep_len = 1;
    atomicMax(max_keep, keep_len);

    const int64_t out_base = static_cast<int64_t>(row) * topk;
    for (int tk = 0; tk < topk; tk++) {
      const bool valid = tk < keep_len;
      route_weight[out_base + tk] = valid ? selected_scores[tk] : 0.0f;
      route_idx[out_base + tk] = valid ? selected_idx[tk] : 0;
      route_mask[out_base + tk] = valid;
    }
  }
}

__global__ void topp_route_scale_kernel(float *__restrict__ route_weight,
                                        int64_t total,
                                        const int *__restrict__ max_keep,
                                        float energy) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (idx < total) {
    route_weight[idx] *= static_cast<float>(max_keep[0]) * energy;
  }
}

__global__ void topp_route_max_keep_nwin7_kernel(
    const float *__restrict__ query,
    int *__restrict__ max_keep,
    int64_t n,
    int64_t qk_dim,
    int64_t topk,
    float p,
    float temperature,
    float scale) {
  __shared__ float s_scores[SPECIAL_P2];
  __shared__ float s_q_norm_acc[ROUTE_THREADS];

  const int row = blockIdx.x;
  const int batch = row / SPECIAL_P2;
  const int q_win = row - batch * SPECIAL_P2;
  const int tid = threadIdx.x;
  const int64_t q_base =
      (static_cast<int64_t>(batch) * SPECIAL_P2 + q_win) * qk_dim;

  float norm_acc = 0.0f;
  for (int64_t d = tid; d < qk_dim; d += blockDim.x) {
    const float qv = query[q_base + d];
    norm_acc += qv * qv;
  }
  s_q_norm_acc[tid] = norm_acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_q_norm_acc[tid] += s_q_norm_acc[tid + stride];
    }
    __syncthreads();
  }
  const float q_inv_norm = rsqrtf(fmaxf(s_q_norm_acc[0], 1.0e-24f));

  if (tid < SPECIAL_P2) {
    const int k_win = tid;
    const int64_t k_base =
        (static_cast<int64_t>(batch) * SPECIAL_P2 + k_win) * qk_dim;
    float dot = 0.0f;
    float k_norm = 0.0f;
    for (int64_t d = 0; d < qk_dim; d++) {
      const float qv = query[q_base + d];
      const float kv = query[k_base + d];
      dot += qv * kv;
      k_norm += kv * kv;
    }
    s_scores[k_win] =
        dot * q_inv_norm * rsqrtf(fmaxf(k_norm, 1.0e-24f)) * scale;
  }
  __syncthreads();

  if (tid == 0) {
    float selected_scores[SPECIAL_MAX_TOPK];
    int selected_idx[SPECIAL_MAX_TOPK];

    for (int tk = 0; tk < topk; tk++) {
      float best = -INFINITY;
      int best_idx = 0;
      for (int col = 0; col < SPECIAL_P2; col++) {
        const float score = s_scores[col];
        bool used = false;
        for (int prev = 0; prev < tk; prev++) {
          if (selected_idx[prev] == col) {
            used = true;
            break;
          }
        }
        if (!used && (score > best || (score == best && col < best_idx))) {
          best = score;
          best_idx = col;
        }
      }
      selected_scores[tk] = best;
      selected_idx[tk] = best_idx;
    }

    float max_score = -INFINITY;
    for (int tk = 0; tk < topk; tk++) {
      max_score = fmaxf(max_score, selected_scores[tk] / temperature);
    }
    float denom = 0.0f;
    for (int tk = 0; tk < topk; tk++) {
      selected_scores[tk] = expf(selected_scores[tk] / temperature - max_score);
      denom += selected_scores[tk];
    }

    float cumsum = 0.0f;
    int keep_len = 0;
    for (int tk = 0; tk < topk; tk++) {
      const float prob = selected_scores[tk] / denom;
      cumsum += prob;
      if (cumsum <= p) {
        keep_len++;
      }
    }
    if (keep_len < 1) keep_len = 1;
    atomicMax(max_keep, keep_len);
  }
}

template <typename scalar_t>
const scalar_t *tensor_data_ptr(torch::Tensor tensor);

template <>
const float *tensor_data_ptr<float>(torch::Tensor tensor) {
  return tensor.data_ptr<float>();
}

template <>
const __half *tensor_data_ptr<__half>(torch::Tensor tensor) {
  return reinterpret_cast<const __half *>(tensor.data_ptr<at::Half>());
}

template <>
const __nv_bfloat16 *tensor_data_ptr<__nv_bfloat16>(torch::Tensor tensor) {
  return reinterpret_cast<const __nv_bfloat16 *>(
      tensor.data_ptr<at::BFloat16>());
}

// 通用路径：保持原有逐查询点计算方式，但直接写 NHWC 输出，省掉还原窗口核。
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
    int64_t n_win,
    int64_t height,
    int64_t width) {
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int64_t global_warp_id =
      static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
  const int64_t total_warps = n * p2 * num_heads * q_len;

  if (global_warp_id >= total_warps) return;

  const int64_t coarse = global_warp_id / (num_heads * q_len);
  const int64_t rem = global_warp_id - coarse * num_heads * q_len;
  const int64_t head = rem / q_len;
  const int64_t q_pos = rem - head * q_len;

  const int64_t batch = coarse / p2;
  const int64_t p = coarse - batch * p2;
  const int64_t head_v = dim / num_heads;
  const int64_t head_q = qk_dim / num_heads;
  const int64_t route_base = coarse * topk;

  float q_local[GENERIC_MAX_ELEMS_PER_LANE] = {0.0f};
  const int q_per_lane = (head_q + WARP_SIZE - 1) / WARP_SIZE;
  const int64_t q_base =
      ((batch * p2 + p) * q_len + q_pos) * qk_dim + head * head_q;
  for (int i = 0; i < q_per_lane && i < GENERIC_MAX_ELEMS_PER_LANE; i++) {
    const int d = lane + i * WARP_SIZE;
    if (d < head_q) {
      q_local[i] = to_float(q_pix[q_base + d]);
    }
  }

  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;
  if (valid_topk < 0) valid_topk = 0;

  const int64_t q_h = height / n_win;
  const int64_t q_w = width / n_win;
  const int64_t win_y = p / n_win;
  const int64_t win_x = p - win_y * n_win;
  const int64_t local_y = q_pos / q_w;
  const int64_t local_x = q_pos - local_y * q_w;
  const int64_t out_h = win_y * q_h + local_y;
  const int64_t out_w = win_x * q_w + local_x;

  const int v_per_lane = (head_v + WARP_SIZE - 1) / WARP_SIZE;
  float o_acc[GENERIC_MAX_ELEMS_PER_LANE] = {0.0f};
  float m_prev = -INFINITY;
  float l_prev = 0.0f;

  for (int64_t tk = 0; tk < valid_topk; tk++) {
    const int64_t kv_window = r_idx[route_base + tk];
    const float route_weight = to_float(r_weight[route_base + tk]);

    for (int64_t kv_pos = 0; kv_pos < kv_len; kv_pos++) {
      const int64_t kv_base =
          ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);

      float partial_score = 0.0f;
      for (int i = 0; i < q_per_lane && i < GENERIC_MAX_ELEMS_PER_LANE; i++) {
        const int d = lane + i * WARP_SIZE;
        if (d < head_q) {
          const float k_val =
              to_float(kv_pix[kv_base + head * head_q + d]) * route_weight;
          partial_score += q_local[i] * k_val;
        }
      }
      const float score = warp_reduce_sum(partial_score) * scale;
      const float m_new = fmaxf(m_prev, score);
      const float old_scale = expf(m_prev - m_new);
      const float cur_scale = expf(score - m_new);

      l_prev = l_prev * old_scale + cur_scale;
      m_prev = m_new;

      for (int i = 0; i < v_per_lane && i < GENERIC_MAX_ELEMS_PER_LANE; i++) {
        const int v_ch = lane + i * WARP_SIZE;
        float v_val = 0.0f;
        if (v_ch < head_v) {
          v_val =
              to_float(kv_pix[kv_base + qk_dim + head * head_v + v_ch]) *
              route_weight;
        }
        o_acc[i] = o_acc[i] * old_scale + cur_scale * v_val;
      }
    }
  }

  for (int i = 0; i < v_per_lane && i < GENERIC_MAX_ELEMS_PER_LANE; i++) {
    const int v_ch = lane + i * WARP_SIZE;
    if (v_ch < head_v) {
      const int64_t out_idx =
          nhwc_offset(batch, out_h, out_w, head * head_v + v_ch,
                      height, width, dim);
      out[out_idx] = valid_topk > 0 ? o_acc[i] / fmaxf(l_prev, 1e-20f) : 0.0f;
    }
  }
}

// 专用路径：一个线程块处理同一窗口、同一头的一组查询点，并分块复用 K/V。
template <typename scalar_t, int QUERY_TILE, int NUM_HEADS>
__global__ void topp_flash_head32_nwin7_kernel(
    const scalar_t *__restrict__ q_pix,
    const scalar_t *__restrict__ kv_pix,
    const scalar_t *__restrict__ r_weight,
    const int64_t *__restrict__ r_idx,
    const int64_t *__restrict__ keep_len,
    float *__restrict__ out,
    int64_t n,
    int64_t q_len,
    int64_t kv_len,
    int64_t topk,
    float scale,
    int64_t height,
    int64_t width,
    int64_t q_tiles) {
  constexpr int CHANNEL_DIM = SPECIAL_HEAD_DIM * NUM_HEADS;
  constexpr int KV_CHANNEL_DIM = CHANNEL_DIM * 2;
  __shared__ int s_idx[SPECIAL_MAX_TOPK];
  __shared__ float s_weight[SPECIAL_MAX_TOPK];
  __shared__ int s_keep;
  __shared__ float s_k[SPECIAL_KV_TILE * SPECIAL_HEAD_DIM];
  __shared__ float s_v[SPECIAL_KV_TILE * SPECIAL_HEAD_DIM];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  const int64_t q_tile = blockIdx.x % q_tiles;
  const int64_t head_block = blockIdx.x / q_tiles;
  const int64_t head = head_block % NUM_HEADS;
  const int64_t coarse = head_block / NUM_HEADS;
  if (coarse >= n * SPECIAL_P2) return;

  const int64_t batch = coarse / SPECIAL_P2;
  const int64_t p = coarse - batch * SPECIAL_P2;
  const int64_t q_pos = q_tile * QUERY_TILE + warp_id;
  const bool q_valid = q_pos < q_len;

  if (threadIdx.x == 0) {
    int64_t keep = keep_len[coarse];
    if (keep > topk) keep = topk;
    if (keep < 0) keep = 0;
    s_keep = static_cast<int>(keep);
  }
  for (int route_col = threadIdx.x; route_col < topk;
       route_col += blockDim.x) {
    const int64_t route_base = coarse * topk + route_col;
    s_idx[route_col] = r_idx[route_base];
    s_weight[route_col] = to_float(r_weight[route_base]);
  }
  __syncthreads();

  const int valid_topk = s_keep;
  const int64_t q_h = height / SPECIAL_N_WIN;
  const int64_t q_w = width / SPECIAL_N_WIN;
  const int64_t win_y = p / SPECIAL_N_WIN;
  const int64_t win_x = p - win_y * SPECIAL_N_WIN;
  const int64_t local_y = q_valid ? q_pos / q_w : 0;
  const int64_t local_x = q_valid ? q_pos - local_y * q_w : 0;
  const int64_t out_h = win_y * q_h + local_y;
  const int64_t out_w = win_x * q_w + local_x;
  const int64_t out_c = head * SPECIAL_HEAD_DIM + lane;

  float q_val = 0.0f;
  if (q_valid) {
    const int64_t q_base =
        ((batch * SPECIAL_P2 + p) * q_len + q_pos) * CHANNEL_DIM +
        head * SPECIAL_HEAD_DIM;
    q_val = to_float(q_pix[q_base + lane]);
  }

  float o_acc = 0.0f;
  float m_prev = -INFINITY;
  float l_prev = 0.0f;

  for (int tk = 0; tk < valid_topk; tk++) {
    const int kv_window = s_idx[tk];
    const float route_weight = s_weight[tk];

    for (int64_t kv_tile = 0; kv_tile < kv_len; kv_tile += SPECIAL_KV_TILE) {
      const int64_t remaining = kv_len - kv_tile;
      const int tile_len =
          static_cast<int>(remaining < SPECIAL_KV_TILE
                               ? remaining
                               : SPECIAL_KV_TILE);
      const int tile_elems = tile_len * SPECIAL_HEAD_DIM;

      for (int elem = threadIdx.x; elem < tile_elems; elem += blockDim.x) {
        const int tile_pos = elem / SPECIAL_HEAD_DIM;
        const int dim_pos = elem - tile_pos * SPECIAL_HEAD_DIM;
        const int64_t kv_base =
            ((batch * SPECIAL_P2 + kv_window) * kv_len +
             (kv_tile + tile_pos)) *
            KV_CHANNEL_DIM;
        s_k[elem] =
            to_float(kv_pix[kv_base + head * SPECIAL_HEAD_DIM + dim_pos]) *
            route_weight;
        s_v[elem] =
            to_float(kv_pix[kv_base + CHANNEL_DIM +
                            head * SPECIAL_HEAD_DIM + dim_pos]) *
            route_weight;
      }
      __syncthreads();

      if (q_valid) {
        for (int tile_pos = 0; tile_pos < tile_len; tile_pos++) {
          const int tile_base = tile_pos * SPECIAL_HEAD_DIM;
          const float partial_score = q_val * s_k[tile_base + lane];
          const float score = warp_reduce_sum(partial_score) * scale;
          const float m_new = fmaxf(m_prev, score);
          const float old_scale = expf(m_prev - m_new);
          const float cur_scale = expf(score - m_new);

          l_prev = l_prev * old_scale + cur_scale;
          o_acc = o_acc * old_scale + cur_scale * s_v[tile_base + lane];
          m_prev = m_new;
        }
      }
      __syncthreads();
    }
  }

  if (q_valid) {
    const int64_t out_idx =
        nhwc_offset(batch, out_h, out_w, out_c, height, width, CHANNEL_DIM);
    out[out_idx] = valid_topk > 0 ? o_acc / fmaxf(l_prev, 1e-20f) : 0.0f;
  }
}

template <int NUM_HEADS>
__global__ void topp_fused_route_flash_head32_nwin7_kernel(
    const float *__restrict__ route_query,
    const float *__restrict__ q_pix,
    const float *__restrict__ kv_pix,
    float *__restrict__ out,
    int64_t n,
    int64_t q_len,
    int64_t kv_len,
    int64_t topk,
    float route_p,
    float route_temperature,
    float route_scale,
    float attn_scale,
    int64_t height,
    int64_t width) {
  constexpr int CHANNEL_DIM = SPECIAL_HEAD_DIM * NUM_HEADS;
  constexpr int KV_CHANNEL_DIM = CHANNEL_DIM * 2;
  __shared__ float s_scores[SPECIAL_P2];
  __shared__ float s_q_norm_acc[THREADS_PER_BLOCK];
  __shared__ int s_idx[SPECIAL_MAX_TOPK];
  __shared__ float s_weight[SPECIAL_MAX_TOPK];
  __shared__ int s_keep;
  __shared__ float s_k[SPECIAL_KV_TILE * SPECIAL_HEAD_DIM];
  __shared__ float s_v[SPECIAL_KV_TILE * SPECIAL_HEAD_DIM];

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int64_t head = blockIdx.x % NUM_HEADS;
  const int64_t coarse = blockIdx.x / NUM_HEADS;
  if (coarse >= n * SPECIAL_P2) return;

  const int64_t batch = coarse / SPECIAL_P2;
  const int64_t p_win = coarse - batch * SPECIAL_P2;
  const int64_t route_q_base =
      (static_cast<int64_t>(batch) * SPECIAL_P2 + p_win) * CHANNEL_DIM;

  float norm_acc = 0.0f;
  for (int64_t d = tid; d < CHANNEL_DIM; d += blockDim.x) {
    const float qv = route_query[route_q_base + d];
    norm_acc += qv * qv;
  }
  s_q_norm_acc[tid] = norm_acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_q_norm_acc[tid] += s_q_norm_acc[tid + stride];
    }
    __syncthreads();
  }
  const float q_inv_norm = rsqrtf(fmaxf(s_q_norm_acc[0], 1.0e-24f));

  if (tid < SPECIAL_P2) {
    const int k_win = tid;
    const int64_t route_k_base =
        (static_cast<int64_t>(batch) * SPECIAL_P2 + k_win) * CHANNEL_DIM;
    float dot = 0.0f;
    float k_norm = 0.0f;
    for (int64_t d = 0; d < CHANNEL_DIM; d++) {
      const float qv = route_query[route_q_base + d];
      const float kv = route_query[route_k_base + d];
      dot += qv * kv;
      k_norm += kv * kv;
    }
    s_scores[k_win] =
        dot * q_inv_norm * rsqrtf(fmaxf(k_norm, 1.0e-24f)) * route_scale;
  }
  __syncthreads();

  if (tid == 0) {
    float selected_scores[SPECIAL_MAX_TOPK];
    int selected_idx[SPECIAL_MAX_TOPK];

    for (int tk = 0; tk < topk; tk++) {
      float best = -INFINITY;
      int best_idx = 0;
      for (int col = 0; col < SPECIAL_P2; col++) {
        const float score = s_scores[col];
        bool used = false;
        for (int prev = 0; prev < tk; prev++) {
          if (selected_idx[prev] == col) {
            used = true;
            break;
          }
        }
        if (!used && (score > best || (score == best && col < best_idx))) {
          best = score;
          best_idx = col;
        }
      }
      selected_scores[tk] = best;
      selected_idx[tk] = best_idx;
    }

    float max_score = -INFINITY;
    for (int tk = 0; tk < topk; tk++) {
      max_score = fmaxf(max_score, selected_scores[tk] / route_temperature);
    }
    float denom = 0.0f;
    for (int tk = 0; tk < topk; tk++) {
      selected_scores[tk] =
          expf(selected_scores[tk] / route_temperature - max_score);
      denom += selected_scores[tk];
    }

    float cumsum = 0.0f;
    int keep_len = 0;
    for (int tk = 0; tk < topk; tk++) {
      const float prob = selected_scores[tk] / denom;
      cumsum += prob;
      if (cumsum <= route_p) {
        keep_len++;
      }
      selected_scores[tk] = prob;
    }
    if (keep_len < 1) keep_len = 1;
    s_keep = keep_len;
    for (int tk = 0; tk < topk; tk++) {
      const bool valid = tk < keep_len;
      s_idx[tk] = valid ? selected_idx[tk] : 0;
      s_weight[tk] = valid ? selected_scores[tk] : 0.0f;
    }
  }
  __syncthreads();

  const int valid_topk = s_keep;
  const int64_t q_h = height / SPECIAL_N_WIN;
  const int64_t q_w = width / SPECIAL_N_WIN;
  const int64_t win_y = p_win / SPECIAL_N_WIN;
  const int64_t win_x = p_win - win_y * SPECIAL_N_WIN;
  const int64_t out_c = head * SPECIAL_HEAD_DIM + lane;

  for (int64_t q_base_pos = 0; q_base_pos < q_len;
       q_base_pos += WARPS_PER_BLOCK) {
    const int64_t q_pos = q_base_pos + warp_id;
    const bool q_valid = q_pos < q_len;
    const int64_t local_y = q_valid ? q_pos / q_w : 0;
    const int64_t local_x = q_valid ? q_pos - local_y * q_w : 0;
    const int64_t out_h = win_y * q_h + local_y;
    const int64_t out_w = win_x * q_w + local_x;

    float q_val = 0.0f;
    if (q_valid) {
      const int64_t q_base =
          ((batch * SPECIAL_P2 + p_win) * q_len + q_pos) * CHANNEL_DIM +
          head * SPECIAL_HEAD_DIM;
      q_val = q_pix[q_base + lane];
    }

    float o_acc = 0.0f;
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    for (int tk = 0; tk < valid_topk; tk++) {
      const int kv_window = s_idx[tk];
      const float route_weight = s_weight[tk];

      for (int64_t kv_tile = 0; kv_tile < kv_len; kv_tile += SPECIAL_KV_TILE) {
        const int64_t remaining = kv_len - kv_tile;
        const int tile_len =
            static_cast<int>(remaining < SPECIAL_KV_TILE
                                 ? remaining
                                 : SPECIAL_KV_TILE);
        const int tile_elems = tile_len * SPECIAL_HEAD_DIM;

        for (int elem = tid; elem < tile_elems; elem += blockDim.x) {
          const int tile_pos = elem / SPECIAL_HEAD_DIM;
          const int dim_pos = elem - tile_pos * SPECIAL_HEAD_DIM;
          const int64_t kv_base =
              ((batch * SPECIAL_P2 + kv_window) * kv_len +
               (kv_tile + tile_pos)) *
              KV_CHANNEL_DIM;
          s_k[elem] = kv_pix[kv_base + head * SPECIAL_HEAD_DIM + dim_pos] *
                      route_weight;
          s_v[elem] =
              kv_pix[kv_base + CHANNEL_DIM +
                     head * SPECIAL_HEAD_DIM + dim_pos] *
              route_weight;
        }
        __syncthreads();

        if (q_valid) {
          for (int tile_pos = 0; tile_pos < tile_len; tile_pos++) {
            const int tile_base = tile_pos * SPECIAL_HEAD_DIM;
            const float partial_score = q_val * s_k[tile_base + lane];
            const float score = warp_reduce_sum(partial_score) * attn_scale;
            const float m_new = fmaxf(m_prev, score);
            const float old_scale = expf(m_prev - m_new);
            const float cur_scale = expf(score - m_new);

            l_prev = l_prev * old_scale + cur_scale;
            o_acc = o_acc * old_scale + cur_scale * s_v[tile_base + lane];
            m_prev = m_new;
          }
        }
        __syncthreads();
      }
    }

    if (q_valid) {
      const int64_t out_idx =
          nhwc_offset(batch, out_h, out_w, out_c, height, width, CHANNEL_DIM);
      out[out_idx] = valid_topk > 0 ? o_acc / fmaxf(l_prev, 1e-20f) : 0.0f;
    }
  }
}

template <typename scalar_t>
void launch_generic(torch::Tensor q_pix,
                    torch::Tensor kv_pix,
                    torch::Tensor r_weight,
                    torch::Tensor r_idx,
                    torch::Tensor keep_len,
                    torch::Tensor out,
                    int64_t num_heads,
                    int64_t qk_dim,
                    int64_t dim,
                    float scale,
                    int64_t n_win,
                    int64_t height,
                    int64_t width,
                    cudaStream_t stream) {
  const int64_t n = q_pix.size(0);
  const int64_t p2 = q_pix.size(1);
  const int64_t q_len = q_pix.size(2);
  const int64_t kv_len = kv_pix.size(2);
  const int64_t topk = r_idx.size(2);
  const int64_t total_warps = n * p2 * num_heads * q_len;
  const int blocks =
      static_cast<int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  topp_flash_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      tensor_data_ptr<scalar_t>(q_pix),
      tensor_data_ptr<scalar_t>(kv_pix),
      tensor_data_ptr<scalar_t>(r_weight),
      r_idx.data_ptr<int64_t>(),
      keep_len.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim, scale,
      n_win, height, width);
}

template <>
void launch_generic<float>(torch::Tensor q_pix,
                           torch::Tensor kv_pix,
                           torch::Tensor r_weight,
                           torch::Tensor r_idx,
                           torch::Tensor keep_len,
                           torch::Tensor out,
                           int64_t num_heads,
                           int64_t qk_dim,
                           int64_t dim,
                           float scale,
                           int64_t n_win,
                           int64_t height,
                           int64_t width,
                           cudaStream_t stream) {
  const int64_t n = q_pix.size(0);
  const int64_t p2 = q_pix.size(1);
  const int64_t q_len = q_pix.size(2);
  const int64_t kv_len = kv_pix.size(2);
  const int64_t topk = r_idx.size(2);
  const int64_t total_warps = n * p2 * num_heads * q_len;
  const int blocks =
      static_cast<int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  topp_flash_kernel<float><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      q_pix.data_ptr<float>(), kv_pix.data_ptr<float>(),
      r_weight.data_ptr<float>(), r_idx.data_ptr<int64_t>(),
      keep_len.data_ptr<int64_t>(), out.data_ptr<float>(),
      n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim, scale,
      n_win, height, width);
}

template <typename scalar_t, int NUM_HEADS, int QUERY_TILE>
void launch_special_tile(torch::Tensor q_pix,
                         torch::Tensor kv_pix,
                         torch::Tensor r_weight,
                         torch::Tensor r_idx,
                         torch::Tensor keep_len,
                         torch::Tensor out,
                         float scale,
                         int64_t height,
                         int64_t width,
                         cudaStream_t stream) {
  const int64_t n = q_pix.size(0);
  const int64_t q_len = q_pix.size(2);
  const int64_t kv_len = kv_pix.size(2);
  const int64_t topk = r_idx.size(2);
  const int64_t q_tiles = (q_len + QUERY_TILE - 1) / QUERY_TILE;
  const int blocks = static_cast<int>(n * SPECIAL_P2 * NUM_HEADS * q_tiles);
  const int threads = QUERY_TILE * WARP_SIZE;

  topp_flash_head32_nwin7_kernel<scalar_t, QUERY_TILE, NUM_HEADS>
      <<<blocks, threads, 0, stream>>>(
          tensor_data_ptr<scalar_t>(q_pix),
          tensor_data_ptr<scalar_t>(kv_pix),
          tensor_data_ptr<scalar_t>(r_weight),
          r_idx.data_ptr<int64_t>(),
          keep_len.data_ptr<int64_t>(),
          out.data_ptr<float>(),
          n, q_len, kv_len, topk, scale, height, width, q_tiles);
}

template <typename scalar_t, int NUM_HEADS>
void launch_special(torch::Tensor q_pix,
                    torch::Tensor kv_pix,
                    torch::Tensor r_weight,
                    torch::Tensor r_idx,
                    torch::Tensor keep_len,
                    torch::Tensor out,
                    float scale,
                    int64_t height,
                    int64_t width,
                    cudaStream_t stream) {
  if (q_pix.size(2) <= 4) {
    launch_special_tile<scalar_t, NUM_HEADS, 4>(
        q_pix, kv_pix, r_weight, r_idx, keep_len, out, scale, height, width,
        stream);
  } else {
    launch_special_tile<scalar_t, NUM_HEADS, WARPS_PER_BLOCK>(
        q_pix, kv_pix, r_weight, r_idx, keep_len, out, scale, height, width,
        stream);
  }
}

template <typename scalar_t>
void dispatch_special(torch::Tensor q_pix,
                      torch::Tensor kv_pix,
                      torch::Tensor r_weight,
                      torch::Tensor r_idx,
                      torch::Tensor keep_len,
                      torch::Tensor out,
                      int64_t num_heads,
                      float scale,
                      int64_t height,
                      int64_t width,
                      cudaStream_t stream) {
  if (num_heads == 2) {
    launch_special<scalar_t, 2>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                                out, scale, height, width, stream);
  } else if (num_heads == 4) {
    launch_special<scalar_t, 4>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                                out, scale, height, width, stream);
  } else if (num_heads == 8) {
    launch_special<scalar_t, 8>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                                out, scale, height, width, stream);
  } else {
    launch_special<scalar_t, 16>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                                 out, scale, height, width, stream);
  }
}

template <int NUM_HEADS>
void launch_fused_route_flash(torch::Tensor route_query,
                              torch::Tensor q_pix,
                              torch::Tensor kv_pix,
                              torch::Tensor out,
                              int64_t topk,
                              float route_p,
                              float route_temperature,
                              float route_scale,
                              float attn_scale,
                              int64_t height,
                              int64_t width,
                              cudaStream_t stream) {
  const int64_t n = q_pix.size(0);
  const int64_t q_len = q_pix.size(2);
  const int64_t kv_len = kv_pix.size(2);
  const int blocks = static_cast<int>(n * SPECIAL_P2 * NUM_HEADS);

  topp_fused_route_flash_head32_nwin7_kernel<NUM_HEADS>
      <<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          route_query.data_ptr<float>(),
          q_pix.data_ptr<float>(),
          kv_pix.data_ptr<float>(),
          out.data_ptr<float>(),
          n, q_len, kv_len, topk, route_p, route_temperature, route_scale,
          attn_scale, height, width);
}

void dispatch_fused_route_flash(torch::Tensor route_query,
                                torch::Tensor q_pix,
                                torch::Tensor kv_pix,
                                torch::Tensor out,
                                int64_t topk,
                                int64_t num_heads,
                                float route_p,
                                float route_temperature,
                                float route_scale,
                                float attn_scale,
                                int64_t height,
                                int64_t width,
                                cudaStream_t stream) {
  if (num_heads == 2) {
    launch_fused_route_flash<2>(
        route_query, q_pix, kv_pix, out, topk, route_p, route_temperature,
        route_scale, attn_scale, height, width, stream);
  } else if (num_heads == 4) {
    launch_fused_route_flash<4>(
        route_query, q_pix, kv_pix, out, topk, route_p, route_temperature,
        route_scale, attn_scale, height, width, stream);
  } else if (num_heads == 8) {
    launch_fused_route_flash<8>(
        route_query, q_pix, kv_pix, out, topk, route_p, route_temperature,
        route_scale, attn_scale, height, width, stream);
  } else {
    launch_fused_route_flash<16>(
        route_query, q_pix, kv_pix, out, topk, route_p, route_temperature,
        route_scale, attn_scale, height, width, stream);
  }
}

bool can_use_specialized_kernel(int64_t p2,
                                int64_t topk,
                                int64_t num_heads,
                                int64_t qk_dim,
                                int64_t dim,
                                int64_t n_win,
                                int64_t height,
                                int64_t width) {
  if (n_win != SPECIAL_N_WIN || p2 != SPECIAL_P2 || topk > SPECIAL_MAX_TOPK) {
    return false;
  }
  if (height % SPECIAL_N_WIN != 0 || width % SPECIAL_N_WIN != 0) {
    return false;
  }
  if (!(num_heads == 2 || num_heads == 4 ||
        num_heads == 8 || num_heads == 16)) {
    return false;
  }
  if (qk_dim != dim || qk_dim % num_heads != 0 || dim % num_heads != 0) {
    return false;
  }
  return qk_dim / num_heads == SPECIAL_HEAD_DIM &&
         dim / num_heads == SPECIAL_HEAD_DIM;
}

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
  const int64_t p2 = q_pix.size(1);
  const int64_t topk = r_idx.size(2);
  TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  TORCH_CHECK(qk_dim % num_heads == 0 && dim % num_heads == 0,
              "qk_dim and dim must be divisible by num_heads");
  TORCH_CHECK(qk_dim / num_heads <=
                  WARP_SIZE * GENERIC_MAX_ELEMS_PER_LANE,
              "qk head dimension is too large for the generic CUDA kernel");
  TORCH_CHECK(dim / num_heads <= WARP_SIZE * GENERIC_MAX_ELEMS_PER_LANE,
              "value head dimension is too large for the generic CUDA kernel");
  auto out = torch::empty({q_pix.size(0), height, width, dim},
                          q_pix.options().dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_special =
      can_use_specialized_kernel(p2, topk, num_heads, qk_dim, dim,
                                 n_win, height, width);

  if (use_special) {
    if (q_pix.scalar_type() == torch::kFloat16) {
      dispatch_special<__half>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                               out, num_heads, static_cast<float>(scale),
                               height, width, stream);
    } else if (q_pix.scalar_type() == torch::kBFloat16) {
      dispatch_special<__nv_bfloat16>(
          q_pix, kv_pix, r_weight, r_idx, keep_len, out, num_heads,
          static_cast<float>(scale), height, width, stream);
    } else {
      dispatch_special<float>(q_pix, kv_pix, r_weight, r_idx, keep_len,
                              out, num_heads, static_cast<float>(scale),
                              height, width, stream);
    }
  } else {
    if (q_pix.scalar_type() == torch::kFloat16) {
      launch_generic<__half>(q_pix, kv_pix, r_weight, r_idx, keep_len, out,
                             num_heads, qk_dim, dim, static_cast<float>(scale),
                             n_win, height, width, stream);
    } else if (q_pix.scalar_type() == torch::kBFloat16) {
      launch_generic<__nv_bfloat16>(
          q_pix, kv_pix, r_weight, r_idx, keep_len, out, num_heads, qk_dim,
          dim, static_cast<float>(scale), n_win, height, width, stream);
    } else {
      launch_generic<float>(q_pix, kv_pix, r_weight, r_idx, keep_len, out,
                            num_heads, qk_dim, dim, static_cast<float>(scale),
                            n_win, height, width, stream);
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor topp_fused_route_flash_forward_cuda(torch::Tensor route_query,
                                                  torch::Tensor q_pix,
                                                  torch::Tensor kv_pix,
                                                  int64_t topk,
                                                  double route_p,
                                                  double route_temperature,
                                                  double route_scale,
                                                  double attn_scale,
                                                  int64_t num_heads,
                                                  int64_t qk_dim,
                                                  int64_t dim,
                                                  int64_t n_win,
                                                  int64_t height,
                                                  int64_t width) {
  TORCH_CHECK(route_query.scalar_type() == torch::kFloat32,
              "fused route query must be float32");
  TORCH_CHECK(q_pix.scalar_type() == torch::kFloat32,
              "fused q_pix must be float32");
  TORCH_CHECK(kv_pix.scalar_type() == torch::kFloat32,
              "fused kv_pix must be float32");
  TORCH_CHECK(route_query.dim() == 3, "route_query must be a 3D tensor");
  TORCH_CHECK(q_pix.dim() == 4, "q_pix must be a 4D tensor");
  TORCH_CHECK(kv_pix.dim() == 4, "kv_pix must be a 4D tensor");
  TORCH_CHECK(route_query.size(0) == q_pix.size(0) &&
                  route_query.size(1) == SPECIAL_P2,
              "route_query shape must be (n, 49, qk_dim)");
  TORCH_CHECK(q_pix.size(0) == kv_pix.size(0) &&
                  q_pix.size(1) == SPECIAL_P2 &&
                  kv_pix.size(1) == SPECIAL_P2,
              "q_pix and kv_pix must have n and p2=49");
  TORCH_CHECK(topk > 0 && topk <= SPECIAL_MAX_TOPK,
              "fused topk must be in [1, 49]");
  TORCH_CHECK(can_use_specialized_kernel(
                  q_pix.size(1), topk, num_heads, qk_dim, dim, n_win, height,
                  width),
              "fused route flash only supports n_win=7, head_dim=32, "
              "heads in {2,4,8,16}, qk_dim==dim");
  TORCH_CHECK(route_query.size(2) == qk_dim &&
                  q_pix.size(3) == qk_dim &&
                  kv_pix.size(3) == qk_dim + dim,
              "fused route flash channel dimensions do not match");
  TORCH_CHECK(q_pix.size(2) == (height / n_win) * (width / n_win),
              "q_len must match the query window area");

  auto out = torch::empty({q_pix.size(0), height, width, dim},
                          q_pix.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();

  dispatch_fused_route_flash(
      route_query, q_pix, kv_pix, out, topk, num_heads,
      static_cast<float>(route_p), static_cast<float>(route_temperature),
      static_cast<float>(route_scale), static_cast<float>(attn_scale), height,
      width, stream);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> topp_route_forward_cuda(torch::Tensor query,
                                                   int64_t topk,
                                                   double p,
                                                   double temperature,
                                                   double energy,
                                                   double scale) {
  TORCH_CHECK(query.scalar_type() == torch::kFloat32,
              "route query must be float32");
  TORCH_CHECK(query.size(1) == SPECIAL_P2, "route query p2 must be 49");
  TORCH_CHECK(topk > 0 && topk <= SPECIAL_MAX_TOPK,
              "route topk must be in [1, 49]");
  const int64_t n = query.size(0);
  const int64_t qk_dim = query.size(2);
  auto route_weight = torch::empty({n, SPECIAL_P2, topk}, query.options());
  auto route_idx = torch::empty(
      {n, SPECIAL_P2, topk}, query.options().dtype(torch::kLong));
  auto route_mask = torch::empty(
      {n, SPECIAL_P2, topk}, query.options().dtype(torch::kBool));
  auto max_keep = torch::zeros({1}, query.options().dtype(torch::kInt));

  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(n * SPECIAL_P2);
  topp_route_nwin7_kernel<<<blocks, ROUTE_THREADS, 0, stream>>>(
      query.data_ptr<float>(),
      route_weight.data_ptr<float>(),
      route_idx.data_ptr<int64_t>(),
      route_mask.data_ptr<bool>(),
      max_keep.data_ptr<int>(),
      n, qk_dim, topk, static_cast<float>(p),
      static_cast<float>(temperature), static_cast<float>(energy),
      static_cast<float>(scale));
  const int64_t route_elems = route_weight.numel();
  const int scale_blocks = static_cast<int>((route_elems + 255) / 256);
  topp_route_scale_kernel<<<scale_blocks, 256, 0, stream>>>(
      route_weight.data_ptr<float>(), route_elems, max_keep.data_ptr<int>(),
      static_cast<float>(energy));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {route_weight, route_idx, route_mask};
}
