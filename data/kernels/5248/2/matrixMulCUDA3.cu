#include "includes.h"
__global__ void matrixMulCUDA3(float *C, float *A, float *B, int n) { int start_row = blockDim.y * blockIdx.y * TILE_WIDTH + threadIdx.y * TILE_WIDTH;  int end_row = start_row + TILE_WIDTH;  int start_col = blockDim.x * blockIdx.x * TILE_WIDTH + threadIdx.x * TILE_WIDTH;  int end_col = start_col + TILE_WIDTH;  for (int row = start_row; row < end_row; row++) { for (int col = start_col; col < end_col; col++) { float C_val = 0;    for (int k = 0; k < n; ++k) { float A_elem = A[row * n + k];     float B_elem = B[k * n + col];     C_val += A_elem * B_elem; }    C[row*n + col] = C_val; } } }