#include "includes.h"
__global__ void applyLinearFunction(int *size, const short *x, short *y, short *a, short *b) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < *size) {
y[ix] = *a + *b * x[ix];
}
}