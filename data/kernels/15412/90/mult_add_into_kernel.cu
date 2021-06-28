#include "includes.h"
__global__ void mult_add_into_kernel(int n, float *a, float *b, float *c)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
c[i] += a[i]*b[i];
}
}