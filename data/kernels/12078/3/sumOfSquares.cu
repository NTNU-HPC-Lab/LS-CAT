#include "includes.h"
__global__ static void sumOfSquares(int* num, int* result, clock_t* time)
{
const int tid = threadIdx.x;
const int size = DATA_SIZE / THREAD_NUM;

int sum = 0;
clock_t start;
if (tid == 0)
start = clock();

for (int i = tid * size; i < (tid + 1) * size; ++i)
{
sum += num[i] * num[i] * num[i];
}

result[tid] = sum;
if (tid == 0)
*time = clock() - start;
}