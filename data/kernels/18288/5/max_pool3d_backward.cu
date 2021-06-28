#include "includes.h"
__global__ void max_pool3d_backward(int B, int N, int M, int C, const int* maxIndex, const float* gradOutput, float* gradInput)
{
for(int i=blockIdx.x;i<B;i+=gridDim.x)
{
for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
{
int c = j%C;
int n = maxIndex[i*M*C+j];
atomicAdd(&gradInput[i*N*C+n*C+c],gradOutput[i*M*C+j]);
}
}
}