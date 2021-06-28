#include "includes.h"
__global__ void mAddDensity(float *dense, float *dense_old, float dt) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
dense[Idx] += dense_old[Idx]*dt;
}