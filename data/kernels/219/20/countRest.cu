#include "includes.h"
__global__ void countRest(int *bin, int *bin_counters, const int num_bins, const int maxBin, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( (xIndex < n) & (bin[xIndex]<num_bins) )
if (bin[xIndex]>= maxBin) atomicAdd(bin_counters+bin[xIndex],1);
}