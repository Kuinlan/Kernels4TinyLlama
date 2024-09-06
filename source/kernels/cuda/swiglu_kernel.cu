// perform swiglu
// usage
//  block size = 128
//  grid size = (size + 128 - 1) / 128
//  extern shared memory size : 2 * block_size * sizeof(float)

__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

