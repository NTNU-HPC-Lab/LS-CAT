#include "includes.h"
// C++ 17 Includes:

// Project Includes:

// Defines:
__global__ void sum_dynamic_kernel(const int* pIn, int* pOut, size_t numInts)
{
extern __shared__ int ps[]; // Automatically points to our shared memory array

// Load shared memory:
ps[threadIdx.x] = pIn[threadIdx.x];
if (threadIdx.x + blockDim.x <  numInts)
ps[threadIdx.x + blockDim.x] = pIn[threadIdx.x + blockDim.x];
if (0 == threadIdx.x && 1 == (1 & numInts))
ps[numInts - 1] = pIn[numInts - 1];

size_t prevNumThreads{numInts};
for (size_t numThreads{blockDim.x}; numThreads > 0; numThreads >>= 1)
{
if (threadIdx.x > numThreads)
return;

__syncthreads();
ps[threadIdx.x] += ps[threadIdx.x + numThreads];
if (1 == (prevNumThreads & 1))
ps[0] += ps[prevNumThreads - 1];

prevNumThreads = numThreads;
}

*pOut = ps[0];
}