#include "includes.h"
__global__ void bottomBoundaryKernel(double* temperature, int block_size) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < block_size) {
temperature[(block_size + 2) * (block_size + 1) + (1 + i)] = 1.0;
}
}