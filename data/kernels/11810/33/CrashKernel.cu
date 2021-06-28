#include "includes.h"
__global__ void CrashKernel (double *array, int nrad, int nsec, int Crash)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
if (array[i*nsec + j] < 0.0)
array[i*nsec + j] = 1.0;
else
array[i*nsec + j] = 0.0;
}
}