#include "includes.h"
__global__ void invalidateFlow_kernel(float *modFlowX, float *modFlowY, const float *constFlowX, const float *constFlowY, int width, int height, float cons_thres) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

if (x < width && y < height) {
int ind = __mul24(y, width) + x;
float mFX = modFlowX[ind];
float mFY = modFlowY[ind];
float cFX = constFlowX[ind];
float cFY = constFlowY[ind];

float err = (mFX - cFX) * (mFX - cFX) + (mFY - cFY) * (mFY - cFY);
err = sqrtf(err);

if (err > cons_thres) {
mFX = nanf("");
mFY = nanf("");
}

modFlowX[ind] = mFX;
modFlowY[ind] = mFY;
}
}