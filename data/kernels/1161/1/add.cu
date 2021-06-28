#include "includes.h"
__global__ void add(int N, double *a,double *b)
{
int tid = blockIdx.x*blockDim.x + threadIdx.x;
if(tid < N)
{
b[tid] = a[tid]*a[tid];
}

}