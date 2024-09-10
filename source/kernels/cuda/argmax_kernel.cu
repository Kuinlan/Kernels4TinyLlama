// Arg Max
// Config:
//   block(512)
//   grid(1)
// Args:
//   input_ptr: Nx1
//   size: N(vocab size)
//   output_idx: 1
__global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  uint32_t tid = threadIdx.x;
  if (tid >= size) {
    return;
  }

  // 每个线程初始化最大值位置，经过循环分别获得各自负责范围的最大值和位置
  size_t max_index = threadIdx.x;
  float max_value = input_ptr[max_index];
  for (size_t i = tid; i < size; i += blockDim.x) {
    if (input_ptr[i] > max_value) {
      max_index = i;
      max_value = input_ptr[i];
    }
  }

  // 使用各自获得的结果和共享内存进行信息交换
  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    // 解决最后一个 warp 规约问题
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;  // 一样大时，取位置在左侧的
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  // 每个线程与自己线程束中的
  warp_reduce_argmax(val, ptr);

  __syncthreads();

  // 每个线程束将它们的最大值写入共享内存
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  // 每个线程负责一个 warp
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

