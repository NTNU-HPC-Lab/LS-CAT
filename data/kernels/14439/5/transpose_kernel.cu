#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void transpose_kernel(double const* matrices, double* transposed) {
int matrix_offset = blockIdx.x * blockDim.x * blockDim.y;
int matrix_index = matrix_offset + blockDim.x * threadIdx.y + threadIdx.x;
int transpose_index = matrix_offset + IDX2C(threadIdx.y, threadIdx.x, blockDim.y);
transposed[transpose_index] = matrices[matrix_index];
}