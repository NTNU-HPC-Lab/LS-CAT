#include "includes.h"
__global__ void sum(int *a, int *b, int *c)
{
int i = blockIdx.x;
c[i] = a[i] + b[i];
}