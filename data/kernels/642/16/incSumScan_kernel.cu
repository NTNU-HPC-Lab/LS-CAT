#include "includes.h"
__global__ void incSumScan_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals)
{
unsigned int tIdx = threadIdx.x;
unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
extern __shared__  unsigned int s_incScan[];
if (gIdx >= numVals) return;

s_incScan[tIdx] = d_inVals[tIdx];
__syncthreads();

for (int offset = 1; offset <= numVals; offset = offset * 2)
{
unsigned int temp = s_incScan[tIdx];
unsigned int neighbor = 0;
if (tIdx >= offset ) {
neighbor = s_incScan[tIdx - offset];
__syncthreads();
s_incScan[tIdx] = temp + neighbor;
}
__syncthreads();
}
d_outVals[tIdx] = s_incScan[tIdx];
}