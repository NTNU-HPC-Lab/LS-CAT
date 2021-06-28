#include "includes.h"
__global__ void MatrixMulKernel(float *d_x, float *d_y, float *d_z, int Width) {

int idx = threadIdx.x;
int idy = threadIdx.y;

float kernelSum = 0;
if ((idx < Width) && (idy < Width)) {
for (int k = 0; k < Width; ++k) {
kernelSum += d_x[idy * Width + k] * d_y[k * Width + idx];
}
d_z[idy * Width + idx] = kernelSum;
}
}