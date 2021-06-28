#include "includes.h"
__global__ void enumsort(int *deva, int *devn)
{
int tid = threadIdx.x;
int i, count=0;

for(i=0;i<N;i++)
if((deva[i]<=deva[tid])&&(i!=tid))
count++;

devn[count]=deva[tid];
}