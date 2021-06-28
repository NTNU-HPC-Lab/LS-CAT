#include "includes.h"
__global__ void multiplyBy2_self(int size, long *inout) {
const int ix = threadIdx.x + blockIdx.x * blockDim.x;

if (ix < size) {
inout[ix] = inout[ix] * 2;
}
}