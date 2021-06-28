#include "includes.h"
__global__ void max_pool3d_forward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount, const float* input, float* output, int* maxIndex)
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
int n = nnIndex[i*M*K+m*K+k];
if (k==0)
{
output[i*M*C+j] = input[i*N*C+n*C+c];
maxIndex[i*M*C+j] = n;
continue;
}

if (input[i*N*C+n*C+c]>output[i*M*C+j])
{
output[i*M*C+j] = input[i*N*C+n*C+c];
maxIndex[i*M*C+j] = n;
}
}
}
}
}