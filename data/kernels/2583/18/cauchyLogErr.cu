#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void cauchyLogErr(int N, int M, float *A, float *Y)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;
int L = N*M;

if (i < N && j < M)
{
// A2 in this case is stored in the doubled rows of A, the length of A is
// doublt that of Y
float a = __expf(A[index+L]);
A[index] = __fmul_rn(fabsf(__fsub_rn(A[index], Y[index])), a);
A[index +L] = -__logf(__fmul_rn(0.5, a)); // stick final sum factor in 2nd part of A so when it sums to total the cost will be correct
}
}