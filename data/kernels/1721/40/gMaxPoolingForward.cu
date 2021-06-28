#include "includes.h"
__global__ void gMaxPoolingForward(float* out, int outRows, int outCols, float* in, int inRows, int inCols, float* mask, int numKernels, int maskCols, int width, int lastWidth) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;

if(tid >= outRows * outCols)
return;

int rowId = tid / outRows;
int colId = tid % outRows;

float* b = in + (rowId * inCols) + (colId * width);
float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;

if(colId == outRows - 1) {
width = lastWidth;
}

float currentMax = b[0] * localMask[0];
for(int i = 1; i < width; ++i) {
if(b[i] * localMask[i] > currentMax) {
currentMax = b[i] * localMask[i];
}
}

out[rowId + (colId * outCols)] = currentMax;
}