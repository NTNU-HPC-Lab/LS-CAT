#include "includes.h"
__global__ void identity(int size, long *in, long *out) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < size) {
out[ix] = in[ix];
}
}