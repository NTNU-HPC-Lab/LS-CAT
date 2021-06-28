#include "includes.h"
__global__ void incSumScanB2_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals, unsigned int* d_blockOffset)
{
//	unsigned int tIdx = threadIdx.x;
unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
extern __shared__  unsigned int s_incScan[];
if (gIdx >= numVals) return;

d_outVals[gIdx] = ( blockIdx.x > 0) ? d_inVals[gIdx] + d_blockOffset[blockIdx.x]: d_inVals[gIdx];

}