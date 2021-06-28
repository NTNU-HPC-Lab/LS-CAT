#include "includes.h"
__global__ void mInitVelocity(float *u_dimX, float *u_dimY) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
u_dimX[Idx] = 0.f;
u_dimY[Idx] = 0.8f/(float)(blockIdx.x+1);
}