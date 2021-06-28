#include "includes.h"
__global__ void vecAdd (int *a, int *b, int *c)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index < N){
c[index] = a[index] + b[index];
}
}