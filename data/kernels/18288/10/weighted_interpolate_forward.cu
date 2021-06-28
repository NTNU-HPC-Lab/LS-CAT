#include "includes.h"
__global__ void weighted_interpolate_forward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount, const float* input, const float* weight, float* output)
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
float w = weight[i*N*K+n*K+k];
output[i*N*C+j] += input[i*M*C+m*C+c]*w;
}
}
}
}