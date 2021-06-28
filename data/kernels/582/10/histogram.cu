#include "includes.h"
__global__ void histogram(const float* d_in, unsigned int* d_out, const float lumMin, const float lumRange, const size_t numBins, const size_t size)
{
int abs_x = threadIdx.x + blockDim.x * blockIdx.x;

if (abs_x > size)
{
return;
}

int bin = (d_in[abs_x] - lumMin) / lumRange * numBins;

//then increment:
atomicAdd(&(d_out[bin]), 1);
}