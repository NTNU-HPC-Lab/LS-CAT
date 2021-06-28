#include "includes.h"
/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/












__global__ void xexp( float* X, float* C, float* Y, float* Z, unsigned int size)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned long int i = idx; i < size; i += stride) {
X[i] = Z[i]*__expf(C[i] - Y[i]);
}
}