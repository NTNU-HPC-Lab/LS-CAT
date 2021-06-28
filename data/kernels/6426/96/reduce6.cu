#include "includes.h"
__global__ void reduce6(const float* g_idata, float* g_odata, float* g_omask, unsigned int n) {
extern __shared__ float sharedData[];
float* sdata = &sharedData[0];
float* smask = &sharedData[blockDim.x];

// perform first level of reduction,
// reading from global memory, writing to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
unsigned int gridSize = blockDim.x * 2 * gridDim.x;
sdata[tid] = 0;
smask[tid] = 0;

// we reduce multiple elements per thread.  The number is determined by the
// number of active thread blocks (via gridSize).  More blocks will result
// in a larger gridSize and therefore fewer elements per thread
while (i < n) {
sdata[tid] += g_idata[i];
smask[tid] += (g_idata[i] > 0 ? 1 : 0);
if (i + blockDim.x < n) {
sdata[tid] += g_idata[i + blockDim.x];
smask[tid] += (g_idata[i + blockDim.x] > 0 ? 1 : 0);
}
i += gridSize;
}
__syncthreads();

// do reduction in shared mem
if (blockDim.x >= 512) {
if (tid < 256) {
sdata[tid] += sdata[tid + 256];
smask[tid] += smask[tid + 256];
}
__syncthreads();
}
if (blockDim.x >= 256) {
if (tid < 128) {
sdata[tid] += sdata[tid + 128];
smask[tid] += smask[tid + 128];
}
__syncthreads();
}
if (blockDim.x >= 128) {
if (tid < 64) {
sdata[tid] += sdata[tid + 64];
smask[tid] += smask[tid + 64];
}
__syncthreads();
}

if (tid < 32) {
if (blockDim.x >= 64) {
sdata[tid] += sdata[tid + 32];
smask[tid] += smask[tid + 32];
}
if (blockDim.x >= 32) {
sdata[tid] += sdata[tid + 16];
smask[tid] += smask[tid + 16];
}
if (blockDim.x >= 16) {
sdata[tid] += sdata[tid + 8];
smask[tid] += smask[tid + 8];
}
if (blockDim.x >= 8) {
sdata[tid] += sdata[tid + 4];
smask[tid] += smask[tid + 4];
}
if (blockDim.x >= 4) {
sdata[tid] += sdata[tid + 2];
smask[tid] += smask[tid + 2];
}
if (blockDim.x >= 2) {
sdata[tid] += sdata[tid + 1];
smask[tid] += smask[tid + 1];
}
}

// write result for this block to global mem
if (tid == 0) {
g_odata[blockIdx.x] = sdata[0];
g_omask[blockIdx.x] = smask[0];
}
}