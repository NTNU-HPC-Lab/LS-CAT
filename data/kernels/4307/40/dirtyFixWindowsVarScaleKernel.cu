#include "includes.h"
__global__ void dirtyFixWindowsVarScaleKernel( float *xMin, float *xMax, float *yMin, float *yMax, const int size, const float h, const float w, const float minWidth) {

int idx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

if (idx < 2*size) {
float paramMin, paramMax;

if (idx < size) {
paramMin = max(-h+1, min(h-1, xMin[idx]));
paramMax = max(-h+1, min(h-1, xMax[idx]));

if (paramMin + minWidth - 0.99 > paramMax) {
const float mean = 0.5 * (paramMin + paramMax);
paramMin = mean - 0.5 * (minWidth - 0.9);
paramMax = mean + 0.5 * (minWidth - 0.9);
}

xMin[idx] = paramMin;
xMax[idx] = paramMax;
} else {
idx -= size;
paramMin = max(-w+1, min(w-1, yMin[idx]));
paramMax = max(-w+1, min(w-1, yMax[idx]));

if (paramMin + minWidth - 0.99 > paramMax) {
const float mean = 0.5 * (paramMin + paramMax);
paramMin = mean - 0.5 * (minWidth - 0.9);
paramMax = mean + 0.5 * (minWidth - 0.9);
}

yMin[idx] = paramMin;
yMax[idx] = paramMax;
}
}
}