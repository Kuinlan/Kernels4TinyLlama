#include <cub/block/block_reduce.cuh>
#include <cstdint>

// RMS Norm
// Config:
//   block(128)
//   grid(1)
// Args:
//   in: 1xN
//   wei: 1xN
//   out: 1xN
//   size: 2048
//   eps: 1e-6


template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  // 无法向量化的部分
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  // block reduce
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();

  // broadcast
  sum = shared_val;

  // 平方和除以总数的平方根，是均值为0的 layernorm
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}
