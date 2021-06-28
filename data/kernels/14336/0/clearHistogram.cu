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
__global__ void clearHistogram(uint *d_Histogram, uint binCount)
{
//clear histogram
for (uint bin = UMAD(blockIdx.x, blockDim.x, threadIdx.x); bin < binCount; bin += UMUL(blockDim.x, gridDim.x))
d_Histogram[bin] = 0;
}