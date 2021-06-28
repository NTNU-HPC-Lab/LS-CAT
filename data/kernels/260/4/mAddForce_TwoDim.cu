#include "includes.h"
__global__ void mAddForce_TwoDim(float *velocityX, float *velocityY, float *forceX, float *forceY, float dt) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
velocityX[Idx] = (velocityX[Idx] >= 0.6)? velocityX[Idx]:velocityX[Idx]+forceX[Idx]*dt;
velocityY[Idx] = (velocityY[Idx] >= 0.6)? velocityY[Idx]:velocityY[Idx]+forceY[Idx]*dt;
}