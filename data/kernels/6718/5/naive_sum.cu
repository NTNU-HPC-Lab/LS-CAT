#include "includes.h"
__device__ void warpReduce(volatile float *sdata, int tid, int bid, int size)
{
if (bid + 32 < size) sdata[tid] += sdata[tid + 32];
if (bid + 16 < size) sdata[tid] += sdata[tid + 16];
if (bid + 8 < size) sdata[tid] += sdata[tid + 8];
if (bid + 4 < size) sdata[tid] += sdata[tid + 4];
if (bid + 2 < size) sdata[tid] += sdata[tid + 2];
if (bid + 1 < size) sdata[tid] += sdata[tid + 1];
}
__global__ void naive_sum(float *input, int size, float *out)
{
const unsigned int tid = threadIdx.x;
const unsigned int bid = blockIdx.x * blockDim.x * 2 + tid;
extern __shared__ float sdata[];
if (!(bid < size)) return;
sdata[tid] = input[bid];
if (bid + blockDim.x < size) sdata[tid] += input[bid + blockDim.x];
__syncthreads();
for (unsigned int offset = blockDim.x/2; offset > 32; offset /= 2) {
if (tid < offset && bid + offset < size) sdata[tid] += sdata[tid + offset];
__syncthreads();
}
if (tid < 32) warpReduce(sdata, tid, bid, size);
if (tid == 0) out[blockIdx.x] = sdata[0];
}