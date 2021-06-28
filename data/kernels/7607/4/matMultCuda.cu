#include "includes.h"
__global__ void matMultCuda(float *cu_C, float *cu_A, float *cu_B, unsigned int n) {

int row = (blockIdx.x * blockDim.x) + threadIdx.x;
int col = (blockIdx.y * blockDim.y) + threadIdx.y;

//Log row and col of each thread
//printf("row : %d , col : %d \n", row, col);

if (row < n && col < n) {
int temp_sum = 0;

for (int elem = 0; elem < n; elem++)
{
temp_sum += cu_A[row * n + elem] * cu_B[elem * n + col];
}

cu_C[row * n + col] = temp_sum;
}
};