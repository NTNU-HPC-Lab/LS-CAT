#include "includes.h"
__global__ void addKernel(int *ic, const int *ia, const int *ib)
{
__syncthreads();
int i = threadIdx.x;
int b = blockIdx.x;
int bd = blockDim.x;
int gd = gridDim.x;

printf("G[%d] B[%d][%d]  t[%d]\n",gd,bd,b,i);
}