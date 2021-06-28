#include "includes.h"
__global__ void applyChanges(float* syn, float* dsyn, int dim, float alpha)
{
int j = blockDim.x*blockIdx.x + threadIdx.x;
int k = blockDim.y*blockIdx.y + threadIdx.y;
syn[j*dim + k] += dsyn[j*dim + k] * alpha;
}