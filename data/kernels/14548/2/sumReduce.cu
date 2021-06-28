#include "includes.h"


#define INTERVALS 1000000

// Max number of threads per block
#define THREADS 512
#define BLOCKS 64

double calculatePiCPU();

// Synchronous error checking call. Enable with nvcc -DDEBUG
__global__ static void sumReduce(int *n, float *g_sum)
{
int tx = threadIdx.x;
__shared__ float s_sum[THREADS];

if (tx < BLOCKS)
s_sum[tx] = g_sum[tx * THREADS];
else
{
s_sum[tx] = 0.0f;
}

// For each block
for (int i = blockDim.x / 2; i > 0; i >>= 1)
{
if (tx < i)
{
s_sum[tx] += s_sum[tx + i];
}
__syncthreads();
}

g_sum[tx] = s_sum[tx];
}