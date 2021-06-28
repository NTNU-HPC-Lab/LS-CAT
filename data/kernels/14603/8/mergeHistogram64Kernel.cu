#include "includes.h"
__global__ void mergeHistogram64Kernel( float *d_Histogram, float *d_PartialHistograms, uint histogramCount )
{
__shared__ uint data[MERGE_THREADBLOCK_SIZE];

uint sum = 0;

for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
{
sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
}

data[threadIdx.x] = sum;

for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
{
__syncthreads();

if (threadIdx.x < stride)
{
data[threadIdx.x] += data[threadIdx.x + stride];
}
}

if (threadIdx.x == 0)
{
d_Histogram[blockIdx.x] = data[0];
}
}