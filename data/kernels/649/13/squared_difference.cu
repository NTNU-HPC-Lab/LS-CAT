#include "includes.h"
__global__ void squared_difference(float *x, float *y, int len) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < len) {
x[i] = (x[i] - y[i])*(x[i] - y[i]);
}
}