#include "includes.h"
__global__ void add(int *a,int *b,int *c)
{
int x = blockIdx.x;
int y = blockIdx.x;
int i = COL*y + x;
c[i] = a[i] + b[i];
}