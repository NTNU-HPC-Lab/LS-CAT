#include "includes.h"
__global__ void count_zero_one(float *vec, float *data, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( (xIndex < n) ){
if (vec[xIndex] == 0)
atomicAdd(data,1);
else if (vec[xIndex] == 1)
atomicAdd(data+1,1);
}
}