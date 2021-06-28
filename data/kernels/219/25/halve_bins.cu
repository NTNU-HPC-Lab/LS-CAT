#include "includes.h"
__global__ void halve_bins(int *bin, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if ( xIndex < n )
bin[xIndex] = bin[xIndex]/2;

}