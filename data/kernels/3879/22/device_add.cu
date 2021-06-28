#include "includes.h"
__global__ void device_add(int *a, int *b, int *c)
{
int blockId = blockIdx.x;
if (blockId < arrSize)
c[blockId] = a[blockId] + b[blockId];
}