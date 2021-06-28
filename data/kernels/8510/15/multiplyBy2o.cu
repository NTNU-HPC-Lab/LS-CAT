#include "includes.h"
__global__ void multiplyBy2o(int *size, const long *in, long *out) {
const int ix = threadIdx.x + blockIdx.x * blockDim.x;

if (ix < *size) {
out[ix] = in[ix] * 2;
}
}