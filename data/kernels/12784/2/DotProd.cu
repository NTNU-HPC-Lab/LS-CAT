#include "includes.h"
__global__ void DotProd(int *a, int *b, int *c) {

__shared__ int temp[THREADS_PER_BLOCK];

int x = threadIdx.x + blockDim.x * blockIdx.x;
printf("Block ID :%d:\n", blockIdx.x);
printf("Block Dim :%d:\n", blockDim.x);
printf("Theard ID :%d:\n", threadIdx.x);
temp[threadIdx.x] = a[x] * b[x];
printf("Temp:%d\n", temp[threadIdx.x]);

__syncthreads();

if (threadIdx.x == 0)
{
int i,sum = 0;
for (i = 0; i < THREADS_PER_BLOCK; i++)
{
sum += temp[i];
}
printf("\nSUM[%d]:%d", blockIdx.x, sum);
atomicAdd(c, sum);
}
}