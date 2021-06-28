#include "includes.h"
__global__ void  register_usage_test(int * results, int size)
{
int gid = blockDim.x * blockIdx.x + threadIdx.x;

int x1 = 3465;
int x2 = 1768;
int x3 = 453;
int x7 = 3465;
int x5 = 1768;
int x6 = 453;
int x4 = x1 + x2 + x3 + x7 + x5 + x6;

if (gid < size)
{
results[gid] =  x4;
}
}