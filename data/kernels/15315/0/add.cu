#include "includes.h"
#define N 100


__global__ void add(int *a, int *c)
{
int tID   = blockIdx.x;
if (tID < N)
{
c[tID] = 3*a[tID];
}
}