#include "includes.h"


__global__ void add(int N, double *a,double *b, double *c)
{
int tid = blockIdx.x*blockDim.x + threadIdx.x;
if(tid < N)
{
c[tid] = a[tid]+b[tid];
}

}