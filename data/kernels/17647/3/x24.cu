#include "includes.h"
__global__ void x24(float* x25, float* x26, float* x27, int x28) {
int x29 = gridDim.x * blockDim.x;
int x30 = threadIdx.x + blockIdx.x * blockDim.x;
while (x30 < x28) {
int x31 = x30;
x27[x31] = x25[x31] * x26[x31];
x30 = x30 + x29;
}
}