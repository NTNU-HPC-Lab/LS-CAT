#include "includes.h"
__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * nx + ix;

if (ix<nx && iy<ny)
{
MatC[idx] = sin(MatA[idx]) + sin(MatB[idx]);
}
}