#include "includes.h"
__global__ void preScan(unsigned int* deviceInput, unsigned int* deviceOutput, int cnt, unsigned int* deviceSum)
{
extern __shared__ unsigned int temp[];
int cntInB = blockDim.x * 2;
int idxInG = cntInB * blockIdx.x + threadIdx.x;

int idxInB = threadIdx.x;
temp[2 * idxInB]		= 0;
temp[2 * idxInB +1]		= 0;

if (idxInG < cnt)
{
temp[idxInB] = deviceInput[idxInG];
}

if (idxInG + blockDim.x < cnt)
{
temp[idxInB + blockDim.x] = deviceInput[idxInG + blockDim.x];
}

int offset = 1;
for (int d = cntInB >> 1; d > 0; d>>=1)
{
__syncthreads();
if (threadIdx.x < d)
{
int ai = offset - 1 + offset * (threadIdx.x * 2);
int bi = ai + offset;
temp[bi] += temp[ai];
}
offset *= 2;
}

__syncthreads();
//before clear the last element, move the last element to deviceSums.
if (threadIdx.x == 0)
{
deviceSum[blockIdx.x] = temp[cntInB - 1];
temp[cntInB - 1] = 0;
}

//downsweep
for (int d = 1; d < cntInB; d *=2)
{
offset >>= 1;
__syncthreads();

if (threadIdx.x < d)
{

int ai = offset - 1 + offset * (threadIdx.x * 2);
int bi = ai + offset;
unsigned int be = temp[bi];
temp[bi] += temp[ai];
temp[ai] = be;
}
}

if (idxInG < cnt)
{
deviceOutput[idxInG] = temp[idxInB];
}

if (idxInG + blockDim.x < cnt)
{
deviceOutput[idxInG + blockDim.x] = temp[idxInB + blockDim.x];
}
}