#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
int i = blockIdx.x;
if(i < N)
c[i] = a[i] + b[i];
}