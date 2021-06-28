#include "includes.h"
__global__ void incSumScanB1_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals, unsigned int* d_blockOffset, unsigned int valOffset)
{
unsigned int tIdx = threadIdx.x;
unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
extern __shared__  unsigned int s_incScan[];
if (gIdx >= numVals) return;

//if it is the first element of a block then we need to add the offset to it.
s_incScan[tIdx] = (tIdx == 0)? d_inVals[gIdx] + valOffset: d_inVals[gIdx];

//	if (tIdx == 0) printf("gIdx =  %d,  d_inVals[ %d ] = %d , s_incScan[ %d ] = %d ,  valOffset = %d .\n", gIdx, gIdx, d_inVals[gIdx], tIdx, s_incScan[tIdx], valOffset);
__syncthreads();

//for (int offset = 1; offset <= numVals; offset = offset * 2)
for (int offset = 1; offset <= blockDim.x; offset = offset * 2)
{
unsigned int temp = s_incScan[tIdx];
unsigned int neighbor = 0;
if (tIdx >= offset) {
neighbor = s_incScan[tIdx - offset];
__syncthreads();
s_incScan[tIdx] = temp + neighbor;
}
__syncthreads();
}
d_outVals[gIdx] = s_incScan[tIdx];

//now set the cumulative sum for this block in the the blockoffsetarray
if ((tIdx + 1) == blockDim.x)
{
if ((blockIdx.x + 1) < gridDim.x)
{
d_blockOffset[blockIdx.x + 1] = s_incScan[tIdx]; //this will still need to be summed with other blocks
}
}
//	if (gIdx < 10 || gIdx > (numVals - 10)) printf("gIdx =  %d,  d_inVals[ %d ] = %d, d_outvals[ %d ] = %d , s_incScan[ %d ] = %d ,  valOffset = %d .\n",
//		 gIdx, gIdx, d_inVals[gIdx], gIdx, d_outVals[gIdx], tIdx, s_incScan[tIdx], valOffset);

}