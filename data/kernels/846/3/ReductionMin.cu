#include "includes.h"




__global__ void ReductionMin(unsigned int *sdata, unsigned int *results, int n)    //take thread divergence into account
{
// extern __shared__ int sdata[];
unsigned int tx = threadIdx.x;

// block-wide reduction
for(unsigned int offset = blockDim.x>>1; offset > 0; offset >>= 1)
{
__syncthreads();
if(tx < offset)
{
if(sdata[tx + offset] < sdata[tx] || sdata[tx] == 0)
sdata[tx] = sdata[tx + offset];
}

}

// finally, thread 0 writes the result
if(threadIdx.x == 0)
{
// the result is per-block
*results = sdata[0];
}
}