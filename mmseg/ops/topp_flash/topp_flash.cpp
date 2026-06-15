#include <torch/extension.h>
#include <pybind11/stl.h>

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
                                                  int64_t width);

torch::Tensor topp_fused_route_flash_forward(torch::Tensor route_query,
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
  TORCH_CHECK(route_query.is_cuda(), "route_query must be a CUDA tensor");
  TORCH_CHECK(q_pix.is_cuda(), "q_pix must be a CUDA tensor");
  TORCH_CHECK(kv_pix.is_cuda(), "kv_pix must be a CUDA tensor");
  TORCH_CHECK(route_query.scalar_type() == torch::kFloat32,
              "route_query must be float32");
  TORCH_CHECK(q_pix.scalar_type() == torch::kFloat32,
              "q_pix must be float32");
  TORCH_CHECK(kv_pix.scalar_type() == torch::kFloat32,
              "kv_pix must be float32");
  return topp_fused_route_flash_forward_cuda(
      route_query.contiguous(), q_pix.contiguous(), kv_pix.contiguous(),
      topk, route_p, route_temperature, route_scale, attn_scale, num_heads,
      qk_dim, dim, n_win, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_forward", &topp_fused_route_flash_forward,
        "PVSA fused topp route and flash forward (inference)");
}
