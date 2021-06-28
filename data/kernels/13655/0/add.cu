#include "includes.h"
__global__ void add(int *a,int *b,int *c)
{
int tID = blockIdx.x;
if(tID<N)
{
c[tID] = a[tID] + b[tID];
}
}