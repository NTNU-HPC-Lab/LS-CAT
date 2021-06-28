#include "includes.h"
__global__ void multiplyBy2_self(int *size, int *in, int *out) {
const int ix = threadIdx.x + blockIdx.x * blockDim.x;

if (ix < *size) {
out[ix] = in[ix] * 2;
in[ix] = out[ix];
}
}