#include "includes.h"
__global__ void add(int* a, int* b, int* c)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int idy = threadIdx.y + blockIdx.y * blockDim.y;

if (idx > WIDTH || idy > HEIGHT) return;

c[idy * WIDTH + idx] = a[idy * WIDTH + idx] + b[idy * WIDTH + idx];
}