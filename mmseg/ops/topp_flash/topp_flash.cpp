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
                                                   torch::Tensor key,
                                                   int64_t topk,
                                                   double p,
                                                   double temperature,
                                                   double energy,
                                                   double scale,
                                                   bool full_route);

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
  return topp_flash_forward_cuda(
      q_pix.contiguous(), kv_pix.contiguous(), r_weight.contiguous(),
      r_idx.contiguous(), keep_len.contiguous(), num_heads, qk_dim, dim,
      scale, n_win, height, width);
}

std::vector<torch::Tensor> topp_route_forward(torch::Tensor query,
                                              torch::Tensor key,
                                              int64_t topk,
                                              double p,
                                              double temperature,
                                              double energy,
                                              double scale,
                                              bool full_route) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  return topp_route_forward_cuda(query.contiguous(), key.contiguous(), topk,
                                 p, temperature, energy, scale, full_route);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topp_flash_forward,
        "PVSA topp flash attention forward (inference)");
  m.def("route_forward", &topp_route_forward,
        "PVSA topp route forward (inference)");
}
