#include "includes.h"
__global__ void mAddExternForce(float *w_dimX, float *w_dimY, float *f_dimX, float *f_dimY, float dt) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
w_dimX[Idx] = -0.5*w_dimX[Idx];
w_dimY[Idx] = -0.5*w_dimY[Idx];
}