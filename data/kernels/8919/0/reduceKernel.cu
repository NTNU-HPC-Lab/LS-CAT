#include "includes.h"

// GPU Libraries

// Macro to handle errors occured in CUDA api
__device__ void recursiveReduce(int *g_inData, int *g_outData, int inSize, int outSize)
{
extern __shared__ int sData[];

// Identification
unsigned int tId = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// Initialize
sData[tId] = 0;

__syncthreads();

// Fill up the shared memory
if (tId < blockDim.x) {
sData[tId] = g_inData[i];
}

__syncthreads();

// Tree based reduction
for (unsigned int d = 1; d < blockDim.x; d *= 2) {
if (tId % (2 * d) == 0)
if (tId + d < blockDim.x)
sData[tId] += sData[tId + d];

__syncthreads();
}

// Write the result for this block to global memory
if (tId == 0) {
g_outData[blockIdx.x] = sData[0];
}


__syncthreads();

// Recursive call
if (outSize > 1 && i == 0) {

// Kernel Launch
recursiveReduce(g_outData, g_outData, outSize, (outSize - 1) / blockDim.x + 1);

}
else return;

}
__global__ void reduceKernel(int *g_inData, int *g_outData, int inSize, int outSize)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i == 0) {
recursiveReduce(g_inData, g_outData, inSize, outSize);
}
}