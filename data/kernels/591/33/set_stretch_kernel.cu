#include "includes.h"
__global__ void set_stretch_kernel(int samps, float mean, float *d_input) {

int t = blockIdx.x * blockDim.x + threadIdx.x;

if (t >= 0 && t < samps)
d_input[t] = mean;
}