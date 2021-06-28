#include "includes.h"
__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length){
int idx = blockDim.x * blockIdx.x + threadIdx.x;

output[idx] = (uint8_t)((input[idx] < UINT8_MAX) * input[idx]) + (input[idx] >= UINT8_MAX) * UINT8_MAX;

__syncthreads();
}