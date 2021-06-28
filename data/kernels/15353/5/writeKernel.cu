#include "includes.h"
__global__ void writeKernel(float* vec, int len)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

if (i >= c_size.x || j >= c_size.y || k >= c_size.z)
return;

for(auto w = 0; w < len; ++w)
{
long int id = w + len * (i + c_size.x * (j + k * c_size.y));
vec[id] = id;
}
}