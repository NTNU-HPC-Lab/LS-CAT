#include "includes.h"


#define N 10


__global__ void add(int *a, int *b, int *c)
{
int tid = blockIdx.x;
if (tid < N)
c[tid] = a[tid] + b[tid];
}