#include "includes.h"
__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * nx + ix;

//printf("nx: %d, ny: %d, ix: %d, iy: %d, idx: %d\n", nx, ny, ix, iy, idx);

if (ix<nx && iy<ny)
{
MatC[idx] = MatA[idx] + MatB[idx];
//printf("GPU Add: %f + %f = %f.\n", MatA[idx], MatB[idx], MatC[idx]);
}
}