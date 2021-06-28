#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void finishAdvX(int N, int M, float *X, float *advX)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

if (i < N && j < M)
{
if (advX[index] < 0)
{
advX[index] = X[index] - 5.0e-5;
}
else if (advX[index] > 0)
{
advX[index] = X[index] + 5.0e-5;
}
else
{
advX[index] = X[index];
}

}
}