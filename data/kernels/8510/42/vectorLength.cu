#include "includes.h"
__global__ void vectorLength(int size, const double *x, const double *y, double *len) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < size) {
len[ix] = sqrt(x[ix] * x[ix] + y[ix] * y[ix]);
}
}