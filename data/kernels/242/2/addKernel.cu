#include "includes.h"
__global__ void addKernel(float *c, float *a, float *b, int size)
{
int i = blockIdx.x * blockDim.x *blockDim.y  + blockDim.x * threadIdx.y * threadIdx.x;
while(i < size)
{
c[i] = a[i] + b[i];
i += gridDim.x * blockDim.x * blockDim.y;
}

}