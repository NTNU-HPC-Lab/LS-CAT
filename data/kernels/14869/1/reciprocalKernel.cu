#include "includes.h"
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
if (idx < vectorSize)
data[idx] = 1.0/data[idx];
}