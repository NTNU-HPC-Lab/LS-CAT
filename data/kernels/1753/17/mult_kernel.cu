#include "includes.h"
__global__ void mult_kernel(float* data, const float scale, const int realtc)
{

const uint index = threadIdx.x + (blockIdx.x + gridDim.x*blockIdx.y)*MAX_THREADS;

if (index < realtc){
data[index] *= scale;
}
}