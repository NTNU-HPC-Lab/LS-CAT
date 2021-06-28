#include "includes.h"
__global__ static void sumOfSquares(int* num, int* result, clock_t* time)
{
//shared memory
extern __shared__ int shared[];
const int tid = threadIdx.x;
const int bid = blockIdx.x;
shared[tid] = 0;

if (tid == 0)
time[bid] = clock();

for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM)
{
shared[tid] += num[i] * num[i] * num[i];
}

//synchronized
__syncthreads();

//sum
if (tid == 0)
{
for (int i = 1; i < THREAD_NUM; ++i)
shared[0] += shared[i];

result[bid] = shared[0];
}

//result[bid * THREAD_NUM + tid] = sum;
if (tid == 0)
time[bid + BLOCK_NUM] = clock();
}