#include <torch/extension.h>

torch::Tensor topp_flash_forward_cuda(torch::Tensor q_pix,
                                      torch::Tensor kv_pix,
                                      torch::Tensor r_weight,
                                      torch::Tensor r_idx,
                                      torch::Tensor r_mask,
                                      int64_t num_heads,
                                      int64_t qk_dim,
                                      int64_t dim,
                                      double scale,
                                      int64_t n_win,
                                      int64_t height,
                                      int64_t width);

torch::Tensor topp_flash_forward(torch::Tensor q_pix,
                                 torch::Tensor kv_pix,
                                 torch::Tensor r_weight,
                                 torch::Tensor r_idx,
                                 torch::Tensor r_mask,
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
  TORCH_CHECK(r_mask.is_cuda(), "r_mask must be a CUDA tensor");
  TORCH_CHECK(q_pix.scalar_type() == torch::kFloat32,
              "pvsa v3.0 CUDA forward currently supports float32 only");
  TORCH_CHECK(kv_pix.scalar_type() == torch::kFloat32,
              "pvsa v3.0 CUDA forward currently supports float32 only");
  TORCH_CHECK(r_weight.scalar_type() == torch::kFloat32,
              "pvsa v3.0 CUDA forward currently supports float32 only");
  TORCH_CHECK(r_idx.scalar_type() == torch::kLong,
              "r_idx must be int64");
  TORCH_CHECK(r_mask.scalar_type() == torch::kBool,
              "r_mask must be bool");

  return topp_flash_forward_cuda(q_pix.contiguous(),
                                 kv_pix.contiguous(),
                                 r_weight.contiguous(),
                                 r_idx.contiguous(),
                                 r_mask.contiguous(),
                                 num_heads,
                                 qk_dim,
                                 dim,
                                 scale,
                                 n_win,
                                 height,
                                 width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topp_flash_forward, "PVSA topp flash forward");
}
