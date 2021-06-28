#include "includes.h"
__global__ void vector_max_kernel(const float* x, int len, int blen, float* result) {
__shared__ float value[256];
int bstart = blen * blockIdx.x;
int start = bstart + threadIdx.x;
int end = min(len, bstart + blen);

float v = 0;
for (int i = start; i < end; i += blockDim.x) v = max(v, fabs(x[i]));
value[threadIdx.x] = v;
// reduce to the first two values
__syncthreads();
if (threadIdx.x < 128)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 128]);
__syncthreads();
if (threadIdx.x < 64)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 64]);
__syncthreads();
if (threadIdx.x < 32)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 32]);
if (threadIdx.x < 16)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 16]);
if (threadIdx.x < 8)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 8]);
if (threadIdx.x < 4)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 4]);
if (threadIdx.x < 2)
value[threadIdx.x] = max(value[threadIdx.x], value[threadIdx.x + 2]);
// write back
if (threadIdx.x == 0) result[blockIdx.x] = max(value[0], value[1]);
}