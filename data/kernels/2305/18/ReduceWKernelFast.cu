#include "includes.h"



__device__ float ReduceFunc(int tid, float* buf)
{
if (tid < 256) {
buf[tid] += buf[tid + 256];
}
__syncthreads();
if (tid < 128) {
buf[tid] += buf[tid + 128];
}
__syncthreads();
if (tid < 64) {
buf[tid] += buf[tid + 64];
}
__syncthreads();
float sum;
if (tid < 32) {
sum = buf[tid] + buf[tid + 32];
sum += __shfl_down_sync(0xffffffff, sum, 16);
sum += __shfl_down_sync(0xffffffff, sum, 8);
sum += __shfl_down_sync(0xffffffff, sum, 4);
sum += __shfl_down_sync(0xffffffff, sum, 2);
sum += __shfl_down_sync(0xffffffff, sum, 1);
}
return sum;
}
__global__ void ReduceWKernelFast(const uint8_t *src, float *dst, int width, int height)
{
int tid = threadIdx.x;
int y = blockIdx.y;

__shared__ float sbuf[512];

float sum = 0;
for (int x = tid; x < width; x += 512) {
sum += src[x + y * width];
}

sbuf[tid] = sum;
__syncthreads();

sum = ReduceFunc(tid, sbuf);

if (tid == 0)
dst[y] = sum;
}