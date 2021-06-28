#include "includes.h"
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int idx = iy * NX + ix;

if (ix < NX && iy < NY)
{
C[idx] = A[idx] + B[idx];
}
}