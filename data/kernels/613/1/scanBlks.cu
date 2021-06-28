#include "includes.h"
//Udacity HW 4
//Radix Sorting





__global__ void scanBlks(unsigned int *in, unsigned int *out, unsigned int n, unsigned int *blkSums)
{

extern __shared__ int blkData[];
int i1 = blockIdx.x * 2 * blockDim.x + threadIdx.x;
int i2 = i1 + blockDim.x;
if (i1 < n)
blkData[threadIdx.x] = in[i1];
if (i2 < n)
blkData[threadIdx.x + blockDim.x] = in[i2];
__syncthreads();


for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
{
int blkDataIdx = (threadIdx.x + 1) * 2 * stride - 1;
if (blkDataIdx < 2 * blockDim.x)
blkData[blkDataIdx] += blkData[blkDataIdx - stride];
__syncthreads();
}

for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
{
int blkDataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
if (blkDataIdx < 2 * blockDim.x)
blkData[blkDataIdx] += blkData[blkDataIdx - stride];
__syncthreads();
}


if (i1 < n)
out[i1] = blkData[threadIdx.x];
if (i2 < n)
out[i2] = blkData[threadIdx.x + blockDim.x];

if (blkSums != NULL && threadIdx.x == 0)
blkSums[blockIdx.x] = blkData[2 * blockDim.x - 1];

}