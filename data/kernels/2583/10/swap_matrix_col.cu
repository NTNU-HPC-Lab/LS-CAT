#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void swap_matrix_col(int N, int C, float *X, float *V)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int index = (C-1)*N + i;

if (i < N)
{
float a = X[index];
X[index] = V[i];
V[i] = a;
}
}