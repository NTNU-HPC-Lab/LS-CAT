#include "includes.h"
__global__ void matrix_2d_mul_float_gpu(float *A, float *B, float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
// Create shared variables (Available to all threads on the same block)
__shared__ float A_tile[N_THREADS][N_THREADS];
__shared__ float B_tile[N_THREADS][N_THREADS];
// Block index
int bx = blockIdx.x; int by = blockIdx.y;
// Thread index
int tx = threadIdx.x; int ty = threadIdx.y;

// Index of the first sub-matrix of A processed by the block
int aBegin = num_cols_A * N_THREADS * by;
// Index of the last sub-matrix of A processed by the block
int aEnd   = aBegin + num_cols_A - 1;
// Index of the first sub-matrix of B processed by the block
int bBegin = N_THREADS * bx;
int bStep  = N_THREADS * num_cols_B;
int aStep  = N_THREADS;

float sum = 0;

for (int a = aBegin, b = bBegin;a <= aEnd;a += aStep, b += bStep) {
A_tile[ty][tx] = A[a + num_cols_A * ty + tx];
B_tile[tx][ty] = B[b + num_cols_B * tx + ty];

// Synchronize to make sure the matrices are loaded
__syncthreads();

for (int k = 0; k < N_THREADS; ++k)
sum += A_tile[ty][k] * B_tile[k][tx];

// Wait other threads to finish their sub-matrices
__syncthreads();
}

// Write the block sub-matrix to device memory;
// each thread writes one element
int c = num_cols_B * N_THREADS * by + N_THREADS * bx;
C[c + num_cols_B * ty + tx] = sum;

}