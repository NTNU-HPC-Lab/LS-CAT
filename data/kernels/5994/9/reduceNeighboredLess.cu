#include "includes.h"
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
// thread id
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// data pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x;
// thread id out of range
if (threadIdx.x >= n) return;
for (int stride = 1; stride < blockDim.x; stride *= 2){
// first data index of this thread
int index = 2 * idx * stride;
// data add
if (index < blockDim.x){
idata[index] += idata[index + stride];
}
__syncthreads();
}
if (threadIdx.x == 0){
g_odata[blockIdx.x] = idata[0];
}
}