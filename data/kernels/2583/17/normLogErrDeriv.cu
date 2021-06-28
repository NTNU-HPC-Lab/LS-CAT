#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void normLogErrDeriv(int N, int M, float *A, float *Y, float *out)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;
int L = N*M;

if (i < N && j < M)
{
// A2 in this case is stored in the doubled rows of A, the length of A is
// doublt that of Y, out is the same length as A and will store both parts of the derivative
float a = __expf(__fmul_rn(2.0, A[index+L]));
float b = __fsub_rn(A[index], Y[index]);
out[index] = __fmul_rn(b, a);
out[index+L] = __fsub_rn(__fmul_rn(out[index], b), 1.0);
}
}