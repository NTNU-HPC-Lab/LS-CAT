#include "includes.h"
__global__ void depthwise_input_backward(int B, int N, int M, int F, int C, int r, int K, const int* nnIndex, const int* nnCount, const int* binIndex, const float* input, const float* filter, const float* gradOutput, float* gradInput)
{
for(int i=blockIdx.x;i<B;i+=gridDim.x)
{
for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
{
int cout = j%(C*r); // output channel ID
int cin = cout/r;   // input channel ID
int m = j/(C*r);    // output point ID
int nnSize = nnCount[i*M+m];

for(int k=0;k<nnSize;k++)
{
int n = nnIndex[i*M*K+m*K+k];   // input point ID
int f = binIndex[i*M*K+m*K+k];

float derIn = gradOutput[i*M*C*r+j]*filter[f*C*r+cout]/nnSize;
atomicAdd(&gradInput[i*N*C+n*C+cin],derIn);
}
}
}
}