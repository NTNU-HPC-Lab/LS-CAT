#include "includes.h"
__global__ void x6(float* x7, float* x8, float* x9, int x10) {
int x11 = gridDim.x * blockDim.x;
int x12 = threadIdx.x + blockIdx.x * blockDim.x;
while (x12 < x10) {
int x13 = x12;
x9[x13] = x7[x13] + x8[x13];
x12 = x12 + x11;
}
}