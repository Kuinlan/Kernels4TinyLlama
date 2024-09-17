#include <cub/block/block_reduce.cuh>
#include <cstdint>

// SGEMV Kernel
// Info:
//   for a Linear(dim_in, dim_out) layer, its weight would be a [dim_in, dim_out] matrix
// Config:
//   block(128)
//   grid(M)
// Args:
//   input: 1xN
//   weight: NxM
//   output: 1xM

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int N) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // 每个 block 处理输出向量一列或者多列的数据
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= N) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    // 使用 128 个线程向量化读取，并行计算点积，将结果放入共享内存
    // 对于 dim 大于 block size 的情况，使用 step 为 blockDim.x 循环操作
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // block level reduce 获取一个向量点积结果
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

// group-wise int8 quant kernel
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      // 进行反量化操作
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}