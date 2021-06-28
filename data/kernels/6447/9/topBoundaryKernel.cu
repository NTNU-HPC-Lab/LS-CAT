#include "includes.h"
__global__ void topBoundaryKernel(double* temperature, int block_size) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < block_size) {
temperature[1 + i] = 1.0;
}
}