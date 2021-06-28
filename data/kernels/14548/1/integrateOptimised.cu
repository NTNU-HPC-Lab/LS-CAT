#include "includes.h"


#define INTERVALS 1000000

// Max number of threads per block
#define THREADS 512
#define BLOCKS 64

double calculatePiCPU();

// Synchronous error checking call. Enable with nvcc -DDEBUG
__global__ void integrateOptimised(int *n, float *g_sum)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int tx = threadIdx.x;

// Shared memory to hold the sum for each block
__shared__ float s_sum[THREADS];

float sum = 0.0f;
float step  = 1.0f / (float)*n;

for (int i = idx + 1; i <= *n; i += blockDim.x * BLOCKS)
{
float x = step * ((float)i - 0.5f);
sum += 4.0f / (1.0f+ x*x);
}
s_sum[tx] = sum * step;

// Wait for all threads to catch up
__syncthreads();

// For each block, do sum using shared memory
for (int i = blockDim.x / 2; i > 0; i >>= 1)
{
if (tx < i)
{
s_sum[tx] += s_sum[tx + i];
}

__syncthreads();
}

// Write results to global memory
g_sum[idx] = s_sum[tx];
}