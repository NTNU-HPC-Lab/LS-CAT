#include "includes.h"
extern "C" {
}
__global__ void reduce_sum_partial(const float* input, float* output, unsigned int len) {
// from http://www.techdarting.com/2014/06/parallel-reduction-in-cuda.html
// Load a segment of the input vector into shared memory
__shared__ float partialSum[2*256];
int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int t = threadIdx.x;
unsigned int start = 2*blockIdx.x*blockDim.x;

if ((start + t) < len)
{
partialSum[t] = input[start + t];
}
else
{
partialSum[t] = 0.0;
}
if ((start + blockDim.x + t) < len)
{
partialSum[blockDim.x + t] = input[start + blockDim.x + t];
}
else
{
partialSum[blockDim.x + t] = 0.0;
}

// Traverse reduction tree
for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
{
__syncthreads();
if (t < stride)
partialSum[t] += partialSum[t + stride];
}
__syncthreads();

// Write the computed sum of the block to the output vector at correct index
if (t == 0 && (globalThreadId*2) < len)
{
output[blockIdx.x] = partialSum[t];
}
}