#include <cub/block/block_reduce.cuh>
#include <cstdint>


// warp this kernel
// usage:
//   grid dim = head number
//   block dim = 128
__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  // 一个 block 负责处理一个头，并且对 之前 的 pos 循环操作 共有 32 个头
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  float* query_head = query + head * head_size;

  // 记录每个头与前面的 key 的对应的头的分数，seq_len 规定了一个最大序列长度
  float* score_head = score_ptr + head * seq_len;
  float scale = 1.f / sqrtf(head_size);

  // group query，多个 query head 对应 一个 key/value head
  int32_t head_offset = (head / kv_mul) * head_size;

  // 每个 thread 处理前面的一个 pos
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
#pragma unroll
    // 对一个 pos 的 一个 head 中的 head_size 做点积
    for (int i = 0; i < head_size; i += 4) {
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      if (i < head_size) {
        score += key_head_float4.x * query_head_float4.x;
      }
      if (i + 1 < head_size) {
        score += key_head_float4.y * query_head_float4.y;
      }
      if (i + 2 < head_size) {
        score += key_head_float4.z * query_head_float4.z;
      }
      if (i + 3 < head_size) {
        score += key_head_float4.w * query_head_float4.w;
      }
    }

    score *= scale;
    // 某个 head 下，Q 与 之前 pos 的 K 的注意力分数
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  head_offset = layer_offset + (head / kv_mul) * head_size;

  // thread 在 head_size 上并行，将 value 乘上对应的 score
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + head_offset + t * kv_dim;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

// thread 在 pos 维度上的并行
__device__ void  softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // 用一个 block 的线程 sweep 所有 pos，找出 block size 个最大值
  float max_val = tid < size ? x[tid] : 0;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  // block reduce
  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();

  // 将最大值从 shared memory 广播至每个线程
  max_val = shared_val;

  // 再次 sweep ，进行 element wise 处理，并且计算分母
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val; // 广播分母

  // 再次 sweep，最后除以分母
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}