#include "includes.h"
__global__ void add(int* a, int *b, int *c)
{
int tid = blockIdx.x; //task ID? this is CPU code so we only got 1, CPU 0
if (tid < N)
c[tid] = a[tid] + b[tid];
}