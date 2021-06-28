#include "includes.h"
__global__ void Add(float *A, int size)
{
const unsigned int numThreads = blockDim.x * gridDim.x;
const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

for (unsigned int i = idx;i < size; i += numThreads)
A[i] = A[i]+ A[i];
}