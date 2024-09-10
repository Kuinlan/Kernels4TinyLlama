// Token Embedding Extraction
// Config:
//   block(128)
//   grid(max prefill sequence length)
// Args:
//   vocab_size: 32000
//   token_num: N
//   weight_dim: 2048
//   input_ptr: Nx1
//   weight_ptr:  vocab_size x weight_dim

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x; // 一个 block 处理一个token
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  // 并行读取 embedding 向量
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

