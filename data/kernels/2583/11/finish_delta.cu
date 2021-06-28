#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void finish_delta(int N, int M, float *A, float *Y, float *out)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
out[index] = copysignf(1.0, __fsub_rn(A[index], Y[index]));

/*
if (A[index] < Y[index])
{
out[index] = -1.0;
}
else if (A[index] > Y[index])
{
out[index] = 1.0;
}
else
{
out[index] = 0.0;
}
*/

}
}