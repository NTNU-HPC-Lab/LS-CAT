#include "includes.h"
__global__ void dot_cmp_kernal_reduce(float *g_idata1, float *g_idata2, float *g_odata)
{
extern __shared__ float sdata[];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = g_idata1[i]*g_idata2[i] + g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x];
__syncthreads();

// do reduction in shared mem
for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}