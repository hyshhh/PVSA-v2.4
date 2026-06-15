#include <torch/extension.h>
#include <pybind11/stl.h>

#include <vector>

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
                                      int64_t width);

std::vector<torch::Tensor> topp_route_forward_cuda(torch::Tensor query,
                                                   int64_t topk,
                                                   double p,
                                                   double temperature,
                                                   double energy,
                                                   double scale);

torch::Tensor topp_flash_forward(torch::Tensor q_pix,
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
  TORCH_CHECK(q_pix.is_cuda(), "q_pix must be a CUDA tensor");
  TORCH_CHECK(kv_pix.is_cuda(), "kv_pix must be a CUDA tensor");
  TORCH_CHECK(r_weight.is_cuda(), "r_weight must be a CUDA tensor");
  TORCH_CHECK(r_idx.is_cuda(), "r_idx must be a CUDA tensor");
  TORCH_CHECK(keep_len.is_cuda(), "keep_len must be a CUDA tensor");
  TORCH_CHECK(r_idx.scalar_type() == torch::kLong, "r_idx must be int64");
  TORCH_CHECK(keep_len.scalar_type() == torch::kLong, "keep_len must be int64");
  TORCH_CHECK(keep_len.dim() == 2, "keep_len must be a 2D tensor");
  TORCH_CHECK(keep_len.size(0) == q_pix.size(0) &&
                  keep_len.size(1) == q_pix.size(1),
              "keep_len shape must match q_pix n and p2");

  auto dtype = q_pix.scalar_type();
  TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
              "q_pix must be float32, float16, or bfloat16");
  TORCH_CHECK(kv_pix.scalar_type() == dtype, "kv_pix dtype must match q_pix");
  TORCH_CHECK(r_weight.scalar_type() == dtype, "r_weight dtype must match q_pix");

  return topp_flash_forward_cuda(q_pix.contiguous(),
                                 kv_pix.contiguous(),
                                 r_weight.contiguous(),
                                 r_idx.contiguous(),
                                 keep_len.contiguous(),
                                 num_heads, qk_dim, dim, scale, n_win, height, width);
}

std::vector<torch::Tensor> topp_route_forward(torch::Tensor query,
                                             int64_t topk,
                                             double p,
                                             double temperature,
                                             double energy,
                                             double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(query.scalar_type() == torch::kFloat32,
              "query must be float32");
  TORCH_CHECK(query.dim() == 3, "query must be a 3D tensor");
  TORCH_CHECK(query.size(1) == 49, "query p2 must be 49");
  TORCH_CHECK(topk > 0 && topk <= 49, "topk must be in [1, 49]");
  return topp_route_forward_cuda(query.contiguous(), topk, p, temperature,
                                 energy, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topp_flash_forward, "PVSA topp flash forward (optimized)");
  m.def("route_forward", &topp_route_forward,
        "PVSA topp route forward (optimized)");
}
