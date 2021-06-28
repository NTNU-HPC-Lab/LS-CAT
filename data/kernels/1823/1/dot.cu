#include "includes.h"
__global__ void dot(int *a, int *b, int *c)
{
/* shared memory cache for partial sum results */
__shared__ int cache[THREADS_PER_BLOCK];

int i = blockIdx.x * blockDim.x + threadIdx.x;
int result = 0;

/* multiplication step: write a partial sum into the cache */
while(i < N)
{
result += a[i] * b[i];
i += blockDim.x * gridDim.x;
}

cache[threadIdx.x] = result;

/* wait for all other threads in the same block */
__syncthreads();

/* reduction step: sum all entries in the cache */
i = blockDim.x / 2;
while (i != 0)
{
/* only threads 0 through i are busy */
if (threadIdx.x < i)
{
cache[threadIdx.x] += cache[threadIdx.x + i];
}

/* wait for all threads within the block */
__syncthreads();

i /= 2;
}

/* thread 0 writes the result for this block */
if (threadIdx.x == 0)
{
c[blockIdx.x] = cache[0];
}
}