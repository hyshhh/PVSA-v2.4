#include "pvsa_topp_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace pvsa_tensorrt {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int SPECIAL_N_WIN = 7;
constexpr int SPECIAL_P2 = SPECIAL_N_WIN * SPECIAL_N_WIN;
constexpr int SPECIAL_HEAD_DIM = 32;
constexpr int SPECIAL_MAX_TOPK = 49;
constexpr int SPECIAL_KV_TILE = 32;
constexpr int ROUTE_THREADS = 128;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x)
{
  return __bfloat162float(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val)
{
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
                                               int64_t dim)
{
  return ((batch * height + h) * width + w) * dim + c;
}

template <int QK_DIM, int BLOCK_THREADS>
__global__ void topp_route_nwin7_kernel_fixed(
    const float *__restrict__ query,
    const float *__restrict__ key,
    float *__restrict__ route_weight,
    int32_t *__restrict__ route_idx,
    int32_t *__restrict__ route_keep_len,
    int64_t n,
    int64_t topk,
    float p,
    float temperature,
    float energy,
    float scale,
    bool full_route)
{
  __shared__ float s_scores[SPECIAL_P2];
  __shared__ float s_q_norm_acc[BLOCK_THREADS];

  const int row = blockIdx.x;
  const int batch = row / SPECIAL_P2;
  const int q_win = row - batch * SPECIAL_P2;
  const int tid = threadIdx.x;
  const int64_t q_base =
      (static_cast<int64_t>(batch) * SPECIAL_P2 + q_win) * QK_DIM;

  if (full_route)
  {
    const int64_t out_base = static_cast<int64_t>(row) * topk;
    for (int tk = tid; tk < topk; tk += BLOCK_THREADS)
    {
      route_weight[out_base + tk] = 1.0f;
      route_idx[out_base + tk] = static_cast<int32_t>(tk);
    }
    if (tid == 0)
    {
      route_keep_len[row] = static_cast<int32_t>(topk);
    }
    return;
  }

  float norm_acc = 0.0f;
  for (int d = tid; d < QK_DIM; d += BLOCK_THREADS)
  {
    const float qv = query[q_base + d];
    norm_acc += qv * qv;
  }
  s_q_norm_acc[tid] = norm_acc;
  __syncthreads();

  for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      s_q_norm_acc[tid] += s_q_norm_acc[tid + stride];
    }
    __syncthreads();
  }
  const float q_inv_norm = rsqrtf(fmaxf(s_q_norm_acc[0], 1.0e-24f));

  if (tid < SPECIAL_P2)
  {
    const int k_win = tid;
    const int64_t k_base =
        (static_cast<int64_t>(batch) * SPECIAL_P2 + k_win) * QK_DIM;
    float dot = 0.0f;
    float k_norm = 0.0f;
#pragma unroll 4
    for (int d = 0; d < QK_DIM; d++)
    {
      const float qv = query[q_base + d];
      const float kv = key[k_base + d];
      dot += qv * kv;
      k_norm += kv * kv;
    }
    s_scores[k_win] =
        dot * q_inv_norm * rsqrtf(fmaxf(k_norm, 1.0e-24f)) * scale;
  }
  __syncthreads();

  // ── 并行 topk 选择 ─────────────────────────────────────────────
  // 用整个 BLOCK_THREADS 并行扫描 49 列找当前最大值，替代原 tid==0
  // 单线程串行 topk×49 次比较。每轮选出 1 个 topk，O(topk * (49/threads
  // + log))，对 BLOCK_THREADS=64 与 SPECIAL_P2=49 几乎纯并行。
  float selected_scores[SPECIAL_MAX_TOPK];
  int selected_idx[SPECIAL_MAX_TOPK];
  __shared__ float s_scores_copy[SPECIAL_P2];
  __shared__ float s_max_val[BLOCK_THREADS];
  __shared__ int s_max_idx[BLOCK_THREADS];

  for (int col = tid; col < SPECIAL_P2; col += BLOCK_THREADS)
  {
    s_scores_copy[col] = s_scores[col];
  }
  __syncthreads();

  for (int tk = 0; tk < topk; tk++)
  {
    // 每线程负责 [tid, tid+BLOCK_THREADS, ...] 范围内的列
    float local_best = -INFINITY;
    int local_best_idx = 0;
    for (int col = tid; col < SPECIAL_P2; col += BLOCK_THREADS)
    {
      const float score = s_scores_copy[col];
      if (score > local_best ||
          (score == local_best && col < local_best_idx))
      {
        local_best = score;
        local_best_idx = col;
      }
    }
    s_max_val[tid] = local_best;
    s_max_idx[tid] = local_best_idx;
    __syncthreads();

    // block 归约找全局最大（等号取小索引保证确定性）
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1)
    {
      if (tid < stride)
      {
        const float a = s_max_val[tid];
        const float b = s_max_val[tid + stride];
        if (b > a || (b == a && s_max_idx[tid + stride] < s_max_idx[tid]))
        {
          s_max_val[tid] = b;
          s_max_idx[tid] = s_max_idx[tid + stride];
        }
      }
      __syncthreads();
    }

    if (tid == 0)
    {
      const float best = s_max_val[0];
      const int best_idx = s_max_idx[0];
      selected_scores[tk] = best;
      selected_idx[tk] = best_idx;
      s_scores_copy[best_idx] = -INFINITY;
    }
    __syncthreads();
  }

  if (tid == 0)
  {

    float max_score = -INFINITY;
    for (int tk = 0; tk < topk; tk++)
    {
      max_score = fmaxf(max_score, selected_scores[tk] / temperature);
    }
    float denom = 0.0f;
    for (int tk = 0; tk < topk; tk++)
    {
      selected_scores[tk] = expf(selected_scores[tk] / temperature - max_score);
      denom += selected_scores[tk];
    }

    float cumsum = 0.0f;
    int keep_len = 0;
    for (int tk = 0; tk < topk; tk++)
    {
      const float prob = selected_scores[tk] / denom;
      cumsum += prob;
      if (cumsum <= p)
      {
        keep_len++;
      }
      selected_scores[tk] = prob;
    }
    if (keep_len < 1)
      keep_len = 1;
    route_keep_len[row] = static_cast<int32_t>(keep_len);

    const int64_t out_base = static_cast<int64_t>(row) * topk;
    for (int tk = 0; tk < topk; tk++)
    {
      const bool valid = tk < keep_len;
      route_weight[out_base + tk] =
          valid ? selected_scores[tk] * energy : 0.0f;
      route_idx[out_base + tk] = valid ? selected_idx[tk] : 0;
    }
  }
}

// 专用路径：一个线程块处理同一窗口、同一头的一组查询点，并分块复用 K/V。
template <typename scalar_t, int QUERY_TILE, int NUM_HEADS>
__global__ void topp_flash_head32_nwin7_kernel(
    const scalar_t *__restrict__ q_pix,
    const scalar_t *__restrict__ kv_pix,
    const scalar_t *__restrict__ r_weight,
    const int32_t *__restrict__ r_idx,
    const int32_t *__restrict__ keep_len,
    float *__restrict__ out,
    int64_t n,
    int64_t q_len,
    int64_t kv_len,
    int64_t topk,
    float scale,
    int64_t height,
    int64_t width,
    int64_t q_tiles,
    bool use_route_weight)
{
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
  if (coarse >= n * SPECIAL_P2)
    return;

  const int64_t batch = coarse / SPECIAL_P2;
  const int64_t p = coarse - batch * SPECIAL_P2;
  const int64_t q_pos = q_tile * QUERY_TILE + warp_id;
  const bool q_valid = q_pos < q_len;

  if (threadIdx.x == 0)
  {
    int32_t keep = keep_len[coarse];
    if (keep > topk)
      keep = topk;
    if (keep < 0)
      keep = 0;
    s_keep = static_cast<int>(keep);
  }
  for (int route_col = threadIdx.x; route_col < topk;
       route_col += blockDim.x)
  {
    const int64_t route_base = coarse * topk + route_col;
    s_idx[route_col] = r_idx[route_base];
    // soft_routing=False 时原始路径的 KVGather 不乘 r_weight，
    // 这里用 1.0 保持一致性（同时避免无效分支，性能无损）。
    s_weight[route_col] =
        use_route_weight ? to_float(r_weight[route_base]) : 1.0f;
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
  if (q_valid)
  {
    const int64_t q_base =
        ((batch * SPECIAL_P2 + p) * q_len + q_pos) * CHANNEL_DIM +
        head * SPECIAL_HEAD_DIM;
    q_val = to_float(q_pix[q_base + lane]);
  }

  float o_acc = 0.0f;
  float m_prev = -INFINITY;
  float l_prev = 0.0f;

  for (int tk = 0; tk < valid_topk; tk++)
  {
    const int kv_window = s_idx[tk];
    const float route_weight = s_weight[tk];

    for (int64_t kv_tile = 0; kv_tile < kv_len; kv_tile += SPECIAL_KV_TILE)
    {
      const int64_t remaining = kv_len - kv_tile;
      const int tile_len =
          static_cast<int>(remaining < SPECIAL_KV_TILE
                               ? remaining
                               : SPECIAL_KV_TILE);
      const int tile_elems = tile_len * SPECIAL_HEAD_DIM;

      for (int elem = threadIdx.x; elem < tile_elems; elem += blockDim.x)
      {
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

      if (q_valid)
      {
        for (int tile_pos = 0; tile_pos < tile_len; tile_pos++)
        {
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

  if (q_valid)
  {
    const int64_t out_idx =
        nhwc_offset(batch, out_h, out_w, out_c, height, width, CHANNEL_DIM);
    out[out_idx] = valid_topk > 0 ? o_acc / fmaxf(l_prev, 1e-20f) : 0.0f;
  }
}

template <int QK_DIM>
void launch_route_batch_nwin7(const float* query,
                              const float* key,
                              float* route_weight,
                              int32_t* route_idx,
                              int32_t* route_keep_len,
                              int64_t n,
                              int64_t topk,
                              float p,
                              float temperature,
                              float energy,
                              float scale,
                              bool full_route,
                              cudaStream_t stream)
{
  const int route_blocks = static_cast<int>(n * SPECIAL_P2);
  constexpr int ROUTE_BLOCK_THREADS = ROUTE_THREADS;
  topp_route_nwin7_kernel_fixed<QK_DIM, ROUTE_BLOCK_THREADS>
      <<<route_blocks, ROUTE_BLOCK_THREADS, 0, stream>>>(
          query, key, route_weight, route_idx, route_keep_len, n, topk,
          p, temperature, energy, scale, full_route);
}

template <int NUM_HEADS, int QUERY_TILE>
void launch_flash_tile(const float* q_pix,
                       const float* kv_pix,
                       const float* r_weight,
                       const int32_t* r_idx,
                       const int32_t* keep_len,
                       float* out,
                       int64_t n,
                       int64_t q_len,
                       int64_t kv_len,
                       int64_t topk,
                       float scale,
                       int64_t height,
                       int64_t width,
                       cudaStream_t stream,
                       bool use_route_weight)
{
  const int64_t q_tiles = (q_len + QUERY_TILE - 1) / QUERY_TILE;
  const int blocks = static_cast<int>(n * SPECIAL_P2 * NUM_HEADS * q_tiles);
  const int threads = QUERY_TILE * WARP_SIZE;
  topp_flash_head32_nwin7_kernel<float, QUERY_TILE, NUM_HEADS>
      <<<blocks, threads, 0, stream>>>(
          q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len, kv_len,
          topk, scale, height, width, q_tiles, use_route_weight);
}

template <int NUM_HEADS>
void launch_flash(const float* q_pix,
                  const float* kv_pix,
                  const float* r_weight,
                  const int32_t* r_idx,
                  const int32_t* keep_len,
                  float* out,
                  int64_t n,
                  int64_t q_len,
                  int64_t kv_len,
                  int64_t topk,
                  float scale,
                  int64_t height,
                  int64_t width,
                  cudaStream_t stream,
                  bool use_route_weight)
{
  if (q_len <= 4) {
    launch_flash_tile<NUM_HEADS, 4>(
        q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len, kv_len,
        topk, scale, height, width, stream, use_route_weight);
  } else {
    launch_flash_tile<NUM_HEADS, WARPS_PER_BLOCK>(
        q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len, kv_len,
        topk, scale, height, width, stream, use_route_weight);
  }
}

bool can_use_specialized_kernel(int64_t p2,
                                int64_t topk,
                                int64_t num_heads,
                                int64_t qk_dim,
                                int64_t dim,
                                int64_t n_win,
                                int64_t height,
                                int64_t width)
{
  if (n_win != SPECIAL_N_WIN || p2 != SPECIAL_P2 || topk <= 0 ||
      topk > SPECIAL_MAX_TOPK) {
    return false;
  }
  if (height % SPECIAL_N_WIN != 0 || width % SPECIAL_N_WIN != 0) {
    return false;
  }
  if (!(num_heads == 2 || num_heads == 4 || num_heads == 8 ||
        num_heads == 16)) {
    return false;
  }
  if (qk_dim != dim || qk_dim % num_heads != 0 || dim % num_heads != 0) {
    return false;
  }
  return qk_dim / num_heads == SPECIAL_HEAD_DIM &&
         dim / num_heads == SPECIAL_HEAD_DIM;
}

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
                         cudaStream_t stream)
{
  if (query == nullptr || key == nullptr || route_weight == nullptr ||
      route_idx == nullptr || route_keep_len == nullptr || n <= 0 ||
      topk <= 0 || topk > SPECIAL_MAX_TOPK ||
      (full_route && topk != SPECIAL_MAX_TOPK) || temperature <= 0.0f ||
      !std::isfinite(p) || !std::isfinite(temperature) ||
      !std::isfinite(energy) || !std::isfinite(scale) || p < 0.0f || p > 1.0f) {
    return cudaErrorInvalidValue;
  }
  if (qk_dim == 64) {
    launch_route_batch_nwin7<64>(
        query, key, route_weight, route_idx, route_keep_len, n, topk, p,
        temperature, energy, scale, full_route, stream);
  } else if (qk_dim == 128) {
    launch_route_batch_nwin7<128>(
        query, key, route_weight, route_idx, route_keep_len, n, topk, p,
        temperature, energy, scale, full_route, stream);
  } else if (qk_dim == 256) {
    launch_route_batch_nwin7<256>(
        query, key, route_weight, route_idx, route_keep_len, n, topk, p,
        temperature, energy, scale, full_route, stream);
  } else if (qk_dim == 512) {
    launch_route_batch_nwin7<512>(
        query, key, route_weight, route_idx, route_keep_len, n, topk, p,
        temperature, energy, scale, full_route, stream);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

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
                         cudaStream_t stream)
{
  const int64_t p2 = SPECIAL_P2;
  const int64_t expected_q_len =
      (height / SPECIAL_N_WIN) * (width / SPECIAL_N_WIN);
  if (q_pix == nullptr || kv_pix == nullptr || r_weight == nullptr ||
      r_idx == nullptr || keep_len == nullptr || out == nullptr || n <= 0 ||
      q_len != expected_q_len || kv_len <= 0 || !std::isfinite(scale) ||
      !can_use_specialized_kernel(p2, topk, num_heads, qk_dim, dim,
                                  n_win, height, width)) {
    return cudaErrorInvalidValue;
  }
  if (num_heads == 2) {
    launch_flash<2>(q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len,
                    kv_len, topk, scale, height, width, stream,
                    use_route_weight);
  } else if (num_heads == 4) {
    launch_flash<4>(q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len,
                    kv_len, topk, scale, height, width, stream,
                    use_route_weight);
  } else if (num_heads == 8) {
    launch_flash<8>(q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len,
                    kv_len, topk, scale, height, width, stream,
                    use_route_weight);
  } else {
    launch_flash<16>(q_pix, kv_pix, r_weight, r_idx, keep_len, out, n, q_len,
                     kv_len, topk, scale, height, width, stream,
                     use_route_weight);
  }
  return cudaGetLastError();
}

}  // namespace pvsa_tensorrt
