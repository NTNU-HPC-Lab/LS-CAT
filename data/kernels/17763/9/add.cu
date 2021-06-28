#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
// use threadIdx.x to access thread index
c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}