#include "includes.h"
__global__ void UniformNormalDistribution(float *from, float *to, int size)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

float tmp;

if (id < size)
{
tmp = normcdf(from[id] * sqrt((float)size));

to[id] = (tmp -0.5)*2;
}
}