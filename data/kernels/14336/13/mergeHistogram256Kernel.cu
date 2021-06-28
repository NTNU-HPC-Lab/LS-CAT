#include "includes.h"
__global__ void mergeHistogram256Kernel( uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount )
{
uint sum = 0;

//for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)//MERGE_THREADBLOCK_SIZE->HISTOGRAM256_BIN_COUNT ??
for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)//original
{
//sum += d_PartialHistograms[blockIdx.x + i * MERGE_THREADBLOCK_SIZE];
sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];//original
}

//__shared__ uint data[HISTOGRAM256_THREADBLOCK_SIZE];
__shared__ uint data[MERGE_THREADBLOCK_SIZE];//original
data[threadIdx.x] = sum;

for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
{
__syncthreads();

//if (threadIdx.x < stride && threadIdx.x + stride < HISTOGRAM256_THREADBLOCK_SIZE)
if (threadIdx.x < stride)//original
{
data[threadIdx.x] += data[threadIdx.x + stride];
}
}

if (threadIdx.x == 0)
{
d_Histogram[blockIdx.x] = data[0];
}
}