#include "includes.h"
__global__ void addtoall(int* a, int b)
{
int i = threadIdx.x;
atomicAdd(&(a[i]), b);
}