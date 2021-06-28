#include "includes.h"
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize)
{
__shared__ float XY[SECTION_SIZE];
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < InputSize) {
XY[threadIdx.x] = X[i];
}

// Perform iterative scan on XY
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
float in;
__syncthreads();
if (threadIdx.x >= stride){
in = XY[threadIdx.x - stride];
}
__syncthreads();
if (threadIdx.x >= stride){
XY[threadIdx.x] += in;
}
}

__syncthreads();
Y[i] = XY[threadIdx.x];
}