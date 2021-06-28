#include "includes.h"

// First solution with global memory

// Shared memory residual calculation
// Reduction code from CUDA Slides - Mark Harris

__global__ void gpu_HeatReduction (float *res, float *result) {

extern __shared__ float sdata[];
unsigned int tid = threadIdx.x;
unsigned int index= blockIdx.x*blockDim.x+ threadIdx.x;

sdata[tid] = res[index];
__syncthreads();


// Reduce the shared table to compute the residual

for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}
if (tid == 0)
{
int blockIndex = blockIdx.x;

result[blockIndex] = sdata[tid];



}

}