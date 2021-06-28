#include "includes.h"
__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] = a[i] + b[i];
}