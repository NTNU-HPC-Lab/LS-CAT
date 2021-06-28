#include "includes.h"
/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/












__global__ void vecInit(float* X, unsigned int size, float value)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < size; i += stride) {
X[i] = value;
}
}