#include "includes.h"
__global__ void rightPackingKernel(double* temperature, double* ghost, int block_size) {
int j = blockDim.x * blockIdx.x + threadIdx.x;
if (j < block_size) {
ghost[j] = temperature[(block_size + 2) * (1 + j) + (block_size)];
}
}