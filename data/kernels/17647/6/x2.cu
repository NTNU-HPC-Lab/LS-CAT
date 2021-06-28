#include "includes.h"
__global__ void x2(float* x3, float x4, int x5) {
int x6 = gridDim.x * blockDim.x;
int x7 = threadIdx.x + blockIdx.x * blockDim.x;
while (x7 < x5) {
x3[x7] = x4;
x7 = x7 + x6;
}
}