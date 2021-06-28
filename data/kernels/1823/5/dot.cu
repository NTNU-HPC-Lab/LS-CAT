#include "includes.h"
__global__ void dot(int *a, int *b, int *temp, int *c)
{
int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
int i = outputIndex;
int result = 0;

/* multiplication step: compute partial sum */
while(i < N)
{
result += a[i] * b[i];
i += blockDim.x * gridDim.x;
}

temp[outputIndex] = result;

/* wait for all threads to be done multiplying */
__syncthreads();

/* reduction step: sum all entries in the block and write to c */
/* this requires that blockDim.x be a power of two! */
i = blockDim.x / 2;
while (i != 0)
{
/* only threads 0 through i are busy */
if (threadIdx.x < i)
{
/* sum our output element with the one half a block away */
temp[outputIndex] += temp[outputIndex + i];
}

/* wait for all threads within the block */
__syncthreads();

i /= 2;
}

/* thread 0 writes the results for this block */
if (threadIdx.x == 0)
{
c[blockIdx.x] = temp[outputIndex];
}
}