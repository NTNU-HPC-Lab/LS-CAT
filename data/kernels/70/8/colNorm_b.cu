#include "includes.h"
/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/












__global__ void colNorm_b( float* X, float* v, float* b, unsigned int size, unsigned int n)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;
unsigned int col;

for (unsigned long int i = idx; i < size; i += stride) {
col = (int)i%n;
X[i] /= v[col]*b[col];
}
}