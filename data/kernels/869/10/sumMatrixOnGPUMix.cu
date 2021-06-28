#include "includes.h"
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = blockIdx.y;
unsigned int idx = iy * nx + ix;

if (ix < nx && iy < ny)
MatC[idx] = MatA[idx] + MatB[idx];
}