#include "includes.h"
__global__ void fillZero(int *c_red, int size)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = id; i < size; i+=stride)
{
c_red[i] = 0;
}
}