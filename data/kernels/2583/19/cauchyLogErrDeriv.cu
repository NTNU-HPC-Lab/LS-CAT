#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void cauchyLogErrDeriv(int N, int M, float *A, float *Y, float *out)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;
int L = N*M;

if (i < N && j < M)
{
float a = __expf(A[index+L]);
if (A[index] > Y[index])
{
out[index] = a;
}
else if (A[index] < Y[index])
{
out[index] = -a;
}
else
{
out[index] = 0.0;
}

out[index+L] = __fmaf_rn(a, fabsf(__fsub_rn(A[index],  Y[index])), -1.0);
// A2 in this case is stored in the doubled rows of A, the length of A is
// doublt that of Y, out is the same length as A and will store both parts of the derivative
}
}