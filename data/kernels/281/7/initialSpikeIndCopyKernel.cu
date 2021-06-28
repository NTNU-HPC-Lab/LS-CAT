#include "includes.h"
__global__ void initialSpikeIndCopyKernel( unsigned short* pLastSpikeInd, const unsigned int noReal)
{
unsigned int globalIndex = threadIdx.x+blockDim.x*blockIdx.x;
unsigned int spikeNo = globalIndex / noReal;
if (globalIndex<noReal*noSpikes)
{
pLastSpikeInd[globalIndex] = pLastSpikeInd[spikeNo*noReal];
}
}