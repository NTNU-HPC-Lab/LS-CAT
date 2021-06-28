#include "includes.h"

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


#define SHARED_MEMORY_SIZE 49152
#define MERGE_THREADBLOCK_SIZE 128

static uint *d_PartialHistograms;

/*
*	Function that maps value to bin in range 0 inclusive to binCOunt exclusive
*/
__global__ void mergePartialHistogramsKernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount,	uint binCount)
{
for (uint bin = blockIdx.x; bin < binCount; bin += gridDim.x)
{
uint sum = 0;
for (uint histogramIndex = threadIdx.x; histogramIndex < histogramCount; histogramIndex += MERGE_THREADBLOCK_SIZE)
{
sum += d_PartialHistograms[bin + histogramIndex * binCount];
}

__shared__ uint data[MERGE_THREADBLOCK_SIZE];
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
d_Histogram[bin] = data[0];
}
}
}