#include "includes.h"
__global__ void mInitForce(float *f_dimX, float *f_dimY) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = (float)threadIdx.x;
float y = (float)blockIdx.x;
float length = sqrt((float)((x-320)*(x-320))+(float)((y-240)*(y-240)));

if(length < SWIRL_RADIUS) {
f_dimX[Idx] = (240.0-y)/length;
f_dimY[Idx] = (x-320.0)/length;
} else {
f_dimX[Idx] = f_dimY[Idx] = 0.f;
}
}