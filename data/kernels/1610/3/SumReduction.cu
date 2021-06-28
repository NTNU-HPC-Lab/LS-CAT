#include "includes.h"
__global__ void SumReduction(int* input, int n)
{
// Handle to thread block group
extern __shared__ int sm[];

// load shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

sm[tid] = (i < n) ? input[i] : 0;

__syncthreads();

// do reduction in shared mem
for (unsigned int s = 1; s < blockDim.x; s *= 2)
{
if ((tid % (2 * s)) == 0)
{
sm[tid] += sm[tid + s];
}

__syncthreads();
}

// write result for this block to global mem
//printf("%d: %d   , block ID: %d \n", threadIdx.x, sm[tid], blockIdx.x);
if (tid == 0) input[blockIdx.x] = sm[0];

}