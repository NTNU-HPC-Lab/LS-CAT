#include "includes.h"
__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n){
// thread id
int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
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

// unrolling in blocks
// 这种优化需要保证blockDim.x为2的k次幂，且最大为1024
if (blockDim.x >= 1024 && threadIdx.x < 512) idata[threadIdx.x] += idata[threadIdx.x + 512];
__syncthreads();
if (blockDim.x >= 512 && threadIdx.x < 256) idata[threadIdx.x] += idata[threadIdx.x + 256];
__syncthreads();
if (blockDim.x >= 256 && threadIdx.x < 128) idata[threadIdx.x] += idata[threadIdx.x + 128];
__syncthreads();
if (blockDim.x >= 128 && threadIdx.x < 64) idata[threadIdx.x] += idata[threadIdx.x + 64];
__syncthreads();

// unrolling sync in thread cluster(stride less than 32)
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