#include "includes.h"
__global__ void initKernel(double* temperature, int block_size) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

if (i < block_size + 2 && j < block_size + 2) {
temperature[(block_size + 2) * j + i] = 0.0;
}
}