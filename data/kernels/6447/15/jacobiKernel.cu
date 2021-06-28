#include "includes.h"
__global__ void jacobiKernel(double* temperature, double* new_temperature, int block_size) {
int i = (blockDim.x * blockIdx.x + threadIdx.x) + 1;
int j = (blockDim.y * blockIdx.y + threadIdx.y) + 1;

if (i <= block_size && j <= block_size) {
new_temperature[j * (block_size + 2) + i] =
(temperature[j * (block_size + 2) + (i - 1)] +
temperature[j * (block_size + 2) + (i + 1)] +
temperature[(j - 1) * (block_size + 2) + i] +
temperature[(j + 1) * (block_size + 2) + i] +
temperature[j * (block_size + 2) + i]) *
DIVIDEBY5;
}
}