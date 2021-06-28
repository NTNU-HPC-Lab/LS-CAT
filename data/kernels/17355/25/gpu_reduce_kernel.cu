#include "includes.h"

__global__ void gpu_reduce_kernel (int N, float * vector, float * sum)
{
extern __shared__ float partialSum[];
int tid = threadIdx.x + blockIdx.x*blockDim.x;

partialSum[threadIdx.x] = 0.f;

__syncthreads();

while(tid < N)
{
partialSum[threadIdx.x] += vector[tid];

tid += blockDim.x*gridDim.x;
}

__syncthreads();

int nTotalThreads = blockDim.x;	/// Total number of active threads

/// Algoritme per calcular la reduccio dels valors actuals a la cache del block
while(nTotalThreads > 1)
{
int halfPoint = (nTotalThreads >> 1);	/// divide by two
/// only the first half of the threads will be active.

if (threadIdx.x < halfPoint)
{
partialSum[threadIdx.x] += partialSum[threadIdx.x + halfPoint];
}

/// imprescindible per les reduccions
__syncthreads();

/// Reducing the binary tree size by two:
nTotalThreads = halfPoint;
}

/// El primer thread del primer block es el k s'encarrega de fer els calculs finals
if(threadIdx.x == 0)
(*sum) = partialSum[0];
}