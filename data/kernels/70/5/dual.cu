#include "includes.h"
/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/












__global__ void dual( float* err, float* Y, float* X, float* Z, unsigned int size)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;
float temp;

err[idx] = 0.0;

for (unsigned int i = idx; i < size; i += stride) {
temp = X[i] - Z[i];
Y[i] += temp;
err[idx] += temp*temp;
}
//    __syncthreads();
}