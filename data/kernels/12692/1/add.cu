#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
//int tid = blockIdx.x; //in N blocks
int tid = threadIdx.x; //in N threads
if(tid<N)
{
c[tid] = a[tid] + b[tid];
}
}