#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void weighting_kernel_transposed(double const* matrices, double const* weights, double* results) {
int grid_index = blockIdx.x * blockDim.x * blockDim.y;
int block_index = blockDim.y * threadIdx.x + threadIdx.y;
int matrix_index = grid_index + block_index;
int weighting_index = blockIdx.x * blockDim.x + threadIdx.x;
results[matrix_index] = matrices[block_index] * weights[weighting_index];
}