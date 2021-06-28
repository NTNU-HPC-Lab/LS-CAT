#include "includes.h"
__global__ void x3(int* x4, int x5, int x6) {
int x7 = gridDim.x * blockDim.x;
int x8 = threadIdx.x + blockIdx.x * blockDim.x;
int x9 = -x5;
while (x8 < x6) {
int x10 = x8;
if (x4[x10] > x5) x4[x10] = x5;
if (x4[x10] < x9) x4[x10] = x9;
x8 = x8 + x7;
}
}