#include "includes.h"
__global__ void count_bins(int *bin, int *bin_counters, const int num_bins, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( (xIndex < n) & (bin[xIndex]<num_bins) )
atomicAdd(bin_counters+bin[xIndex],1);
}