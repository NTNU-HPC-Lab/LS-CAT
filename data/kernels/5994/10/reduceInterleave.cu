#include "includes.h"
__global__ void reduceInterleave(int *g_idata, int *g_odata, unsigned int n){
// thread id
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// data pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x;
// thread id out of range
if (idx >= n) return;
for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
if (threadIdx.x < stride){
idata[threadIdx.x] += idata[threadIdx.x + stride];
}
__syncthreads();
}
if (threadIdx.x == 0){
g_odata[blockIdx.x] = idata[0];
}
}