#include "includes.h"
__global__ void x15(float* x16, float* x17, float* x18, int x19) {
int x20 = gridDim.x * blockDim.x;
int x21 = threadIdx.x + blockIdx.x * blockDim.x;
while (x21 < x19) {
int x22 = x21;
x18[x22] = x16[x22] - x17[x22];
x21 = x21 + x20;
}
}