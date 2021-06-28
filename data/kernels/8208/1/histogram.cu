#include "includes.h"
__global__ void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {

//@@ Using privitization technique
__shared__ unsigned int hist[NUM_BINS];

int numOfElementsPerThread = NUM_BINS / BLOCK_SIZE;

int i = blockDim.x * blockIdx.x + threadIdx.x;

for (int j = 0; j < numOfElementsPerThread; ++j)
hist[threadIdx.x + blockDim.x*j] = 0;

__syncthreads();

if (i < num_elements)
atomicAdd(&hist[input[i]], 1);
__syncthreads();

for (int k = 0; k < numOfElementsPerThread; ++k)
atomicAdd(&bins[threadIdx.x + blockDim.x*k], hist[threadIdx.x+blockDim.x*k]);
}