#include "includes.h"
__global__ void fillarray_kernel(float *x, float v, int np) {
int ii = threadIdx.x + blockIdx.x * BLOCKSIZE;
while (ii < np) {
x[ii] = v;
ii += BLOCKSIZE * gridDim.x; //grid strides
}
}