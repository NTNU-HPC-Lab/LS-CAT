#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void absErr(int N, int M, float *A, float *Y)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
A[index] = fabsf(__fsub_rn(A[index], Y[index]));
// A[index] = abs(A[index]-Y[index])
}
}