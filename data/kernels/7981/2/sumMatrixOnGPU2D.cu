#include "includes.h"
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, const int nx, const int ny){
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
int idx = iy * nx + ix;
if(ix < nx && iy < ny)
C[idx] = A[idx] + B[idx];
}