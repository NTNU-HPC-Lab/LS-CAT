#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void weighting_kernel (double const* matrices, double const* weights, double* results) {
int matrix_grid_index = blockIdx.x * blockDim.x * blockDim.y;
int block_index = blockDim.y * threadIdx.x + threadIdx.y;
int matrix_index = matrix_grid_index + block_index;
int weight_index = blockIdx.x * blockDim.y + threadIdx.y;
results[matrix_index] = matrices[block_index] * weights[weight_index];
}