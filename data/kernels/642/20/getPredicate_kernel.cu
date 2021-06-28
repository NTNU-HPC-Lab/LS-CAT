#include "includes.h"
__global__ void getPredicate_kernel(unsigned int * d_inVal, unsigned int * d_predVal, unsigned int numElems, unsigned int bitMask)
{

unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

if (gIdx < numElems)
{
// if bitmask matches inputvale then assign 1 to the position otherwise set to 0
// we'll need to run an inclusive scan later to get the position
d_predVal[gIdx] = ((d_inVal[gIdx] & bitMask) == bitMask) ? 1 : 0;
//d_npredVal[gIdx] = ((d_inVal[gIdx] & bitMask) == bitMask) ? 0 : 1;
}
}