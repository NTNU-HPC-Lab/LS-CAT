#include "includes.h"
__global__ void matrixMultiKernel(float *C, float *A, float *B, int Width) {

const int BLOCK_SIZE = 16; // NOTE: This must be similar to line 338
// block indexes
int bx = blockIdx.x;
int by = blockIdx.y;

// thread indexes
int tx = threadIdx.x;
int ty = threadIdx.y;

// int col = bx * TILE_WIDTH  + tx
// int row = by * TILE_WIDTH  + ty

// Dividing the matrices into sub sections
// Dividing the matrix A
int a_begin = Width * BLOCK_SIZE * by;
int a_end = a_begin + Width - 1;
int a_step = BLOCK_SIZE;

// Dividing the matrix B
int b_begin = BLOCK_SIZE * bx;
int b_step = BLOCK_SIZE * Width;

float temp_c = 0;

// loop throught the submatrices
for (int a = a_begin, b = b_begin; a <= a_end;
a += a_step, b += b_step) {
// sub matrices
__shared__ float sub_a[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float sub_b[BLOCK_SIZE][BLOCK_SIZE];

sub_a[ty][tx] = A[a + Width * ty + tx];
sub_b[ty][tx] = A[b + Width * ty + tx];

__syncthreads();


// loop unroll may not work on cuda if compilation level -O3
// effects cuda code as wll in the assignment
// sub matrix multiplication
#pragma unroll
for (int k = 0; k < BLOCK_SIZE; ++k) {
temp_c += sub_a[ty][k] * sub_b[k][tx];
}
// sync all the global threads running the computations
__syncthreads();
}
int c = Width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + Width * ty + tx] = temp_c;
//    printf("kernel Done \n");
}