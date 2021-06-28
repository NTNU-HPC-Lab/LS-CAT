#include "includes.h"
__global__ void addKernel(int *c, const int *a, const int *b)
{
int g_tId = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int warpId = threadIdx.x / warpSize;

if ( g_tId < vectorSize) {
c[g_tId] = a[g_tId] + b[g_tId];
printf("thread id %d , warp Id %d , block id %d\n", g_tId, warpId,blockIdx.x);
}
}