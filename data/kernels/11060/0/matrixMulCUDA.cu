#include "includes.h"
// System includes

// CUDA runtime

// Helper functions and utilities to work with CUDA


//template <int BLOCK_SIZE> __global__ void




uint32_t h_C[169] = { 0 };




__global__ void matrixMulCUDA(int *A, int *B, int *C)
{
//const int BLOCK_SIZE = 13;
// Block index
//int bx = blockIdx.x;
//int by = blockIdx.y;

// Thread index
int row = threadIdx.x;
int col = threadIdx.y;

int multi = 0;

for (int j = 0; j < 13; j++) {
multi += A[(row * 13) + j] * B[col + (13 * j)];
}
__syncthreads();
C[(row*13)+col] = multi + A[(row * 13) + col] + B[(row * 13)+col];
}