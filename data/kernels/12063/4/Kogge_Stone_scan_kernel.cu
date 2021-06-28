#include "includes.h"
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize)
{
__shared__ float XY[SECTION_SIZE];
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < InputSize && threadIdx.x != 0) {
XY[threadIdx.x] = X[i - 1];
}
else {
XY[threadIdx.x] = 0;
}

if (threadIdx.x < InputSize)
{
// Perform iterative exclusive scan on XY
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
if (threadIdx.x >= stride) {
__syncthreads();
XY[threadIdx.x] += XY[threadIdx.x - stride];
}
}
Y[i] = XY[threadIdx.x];
}
}