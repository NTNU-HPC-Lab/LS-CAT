#include "includes.h"
__global__ void ladKernel(float *a, float *b, float *out, int size) {
extern __shared__ float sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
int stride = blockDim.x * 2 * gridDim.x;
sdata[tid] = 0;
while (i < size) {
sdata[tid] += abs(a[i] - b[i]) + abs(a[i + blockDim.x] - b[i + blockDim.x]);
i += stride;
__syncthreads();
}
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
if (tid<s)
sdata[tid] += sdata[tid + s];
__syncthreads();

}
if (tid < 32) {
sdata[tid] += sdata[tid + 32];
__syncthreads();
sdata[tid] += sdata[tid + 16];
__syncthreads();
sdata[tid] += sdata[tid + 8];
__syncthreads();
sdata[tid] += sdata[tid + 4];
__syncthreads();
sdata[tid] += sdata[tid + 2];
__syncthreads();
sdata[tid] += sdata[tid + 1];
__syncthreads();
}
if (tid == 0) {
out[blockIdx.x] = sdata[0];
}
}