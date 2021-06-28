#include "includes.h"
__global__ void x33(float* x34, float* x35, float* x36, int x37) {
int x38 = gridDim.x * blockDim.x;
int x39 = threadIdx.x + blockIdx.x * blockDim.x;
while (x39 < x37) {
int x40 = x39;
x36[x40] = x34[x40] / x35[x40];
x39 = x39 + x38;
}
}