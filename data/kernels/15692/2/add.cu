#include "includes.h"
__global__ void add(int* a, int* b, int* c)
{
int id = threadIdx.x;
if (id < N)
{
c[id] = b[id] + a[id];
}
}