#include "includes.h"
/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/












__global__ void rowNorm_a( float* X, float* v, float* a, unsigned int size, unsigned int n)
{
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;
unsigned int row;

for (unsigned long int i = idx; i < size; i += stride) {
row = (int)i/n;
X[i] /= v[row]*a[row];
}
}