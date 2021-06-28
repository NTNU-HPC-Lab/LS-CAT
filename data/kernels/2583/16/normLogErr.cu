#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void normLogErr(int N, int M, float *A, float *Y)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;
int L = N*M;

if (i < N && j < M)
{
// A2 in this case is stored in the doubled rows of A, the length of A is
// doublt that of Y
float a = __expf(__fmul_rn(2.0, A[index+L]));
A[index] = __fmul_rn(a, __fmaf_rn(0.5, __fmul_rn(Y[index], Y[index]), __fsub_rn(__fmul_rn(0.5, __fmul_rn(A[index], A[index])),  __fmul_rn(A[index], Y[index]))));
A[index+L] = __fsub_rn(0.9189385332, A[index+L]); // stick final sum factor in 2nd part of A so when it sums to total the cost will be correct
// A[index] = a*(A[index]*(0.5*A[index] - Y[index]) + 0.5*Y[index]*Y[index]);
// A[index+L] = __fsub_rn(0.9189385332, A[index+L]);
}
}