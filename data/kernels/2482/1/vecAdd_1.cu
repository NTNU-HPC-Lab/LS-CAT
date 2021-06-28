#include "includes.h"
__global__ void vecAdd_1(double *a, double *b, double *c, int n)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < n)
{
c[id] = a[id] + b[id];
}

}