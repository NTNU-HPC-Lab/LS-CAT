#include "includes.h"
__global__ void depthwise_filter_backward(int B, int N, int M, int F, int C, int r, int K, const int* nnIndex, const int* nnCount, const int* binIndex, const float* input, const float* gradOutput, float* gradFilter, int sharedMemSize, int startIdx)
{
extern __shared__ float gradPerBlock[]; // the gradient on each block
for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
{
gradPerBlock[i] = 0; // for 1D block
}
__syncthreads();

int endIdx = sharedMemSize+startIdx;
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

float derFilt = gradOutput[i*M*C*r+j]*input[i*N*C+n*C+cin]/nnSize;

int currIdx = f*C*r+cout;
if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
{
atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt);
}
}
}
}
__syncthreads();

for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
{
atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
}
}