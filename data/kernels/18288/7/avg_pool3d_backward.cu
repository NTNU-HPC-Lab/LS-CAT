#include "includes.h"
__global__ void avg_pool3d_backward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount, const float* gradOutput, float* gradInput)
{
for(int i=blockIdx.x;i<B;i+=gridDim.x)
{
for(int j=threadIdx.x;j<M*C;j+=blockDim.x)
{
int m = j/C;
int c = j%C;
int nnSize = nnCount[i*M+m];
for(int k=0;k<nnSize;k++)
{
int n = nnIndex[i*M*K+m*K+k]; // only neighbor, no bin indices, dimension=(B,M,K)
atomicAdd(&gradInput[i*N*C+n*C+c],gradOutput[i*M*C+j]/nnSize);
}
}
}
}