#include "includes.h"
__global__ void mAddDrip(float *dense, int centerX, int centerY, float redius) {
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
int x = threadIdx.x;
int y = blockIdx.x;
float length = sqrt((float)((x-centerX)*(x-centerX))+(float)((y-centerY)*(y-centerY)));

if(length < redius) {
dense[Idx] += 200;
}
}