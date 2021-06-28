#include "includes.h"
__global__ void saturate(unsigned int *bins, unsigned int num_bins) {

//@@If the bin value is more than 127, make it equal to 127
for (int i = 0; i < NUM_BINS / BLOCK_SIZE; ++i)

if (bins[threadIdx.x + blockDim.x*i] >= 128)

bins[threadIdx.x + blockDim.x*i]  = 127;
}