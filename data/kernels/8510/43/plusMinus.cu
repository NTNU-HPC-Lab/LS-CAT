#include "includes.h"
__global__ void plusMinus(int size, const double *base, const float *deviation, double *a, float *b) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < size) {
a[ix] = base[ix] - deviation[ix];
b[ix] = base[ix] + deviation[ix];
}
}