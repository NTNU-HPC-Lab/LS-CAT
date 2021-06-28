#include "includes.h"
__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n){
// thread id
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// data pointer of this block(s)
int *idata = g_idata + blockIdx.x * blockDim.x * 8;
// unrolling blocks
if (idx + 7 * blockDim.x < n) {
int el0 = g_idata[idx];
int el1 = g_idata[idx + blockDim.x];
int el2 = g_idata[idx + 2*blockDim.x];
int el3 = g_idata[idx + 3*blockDim.x];
int el4 = g_idata[idx + 4*blockDim.x];
int el5 = g_idata[idx + 5*blockDim.x];
int el6 = g_idata[idx + 6*blockDim.x];
int el7 = g_idata[idx + 7*blockDim.x];
g_idata[idx] = el0+el1+el2+el3+el4+el5+el6+el7;
}
__syncthreads();
// thread id out of range
if (idx >= n) return;
for (int stride = blockDim.x/2; stride > 32; stride >>= 1){
if (threadIdx.x < stride){
idata[threadIdx.x] += idata[threadIdx.x + stride];
}
__syncthreads();
}
// unrolling sync in blocks(stride less than 32)
if (threadIdx.x < 32){
volatile int *vmem = idata;
vmem[threadIdx.x] += vmem[threadIdx.x + 32];
vmem[threadIdx.x] += vmem[threadIdx.x + 16];
vmem[threadIdx.x] += vmem[threadIdx.x + 8];
vmem[threadIdx.x] += vmem[threadIdx.x + 4];
vmem[threadIdx.x] += vmem[threadIdx.x + 2];
vmem[threadIdx.x] += vmem[threadIdx.x + 1];
}
if (threadIdx.x == 0){
g_odata[blockIdx.x] = idata[0];
}
}