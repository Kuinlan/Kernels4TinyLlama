#include <cstdint>

// warp this kernel
// Add operation for residual add
// usage:
//   block size = 512
//   grid size = (total_num + block_size - 1) / block_size
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

