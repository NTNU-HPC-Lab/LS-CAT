#include "includes.h"
__global__ void mean_interpolate_backward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount, const float* gradOutput, float* gradInput)
{
for(int i=blockIdx.x;i<B;i+=gridDim.x)
{
for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
{
int n = j/C;
int c = j%C;
int nnSize = nnCount[i*N+n];
for(int k=0;k<nnSize;k++)
{
int m = nnIndex[i*N*K+n*K+k];
atomicAdd(&gradInput[i*M*C+m*C+c],gradOutput[i*N*C+j]/nnSize);
}
}
}
}