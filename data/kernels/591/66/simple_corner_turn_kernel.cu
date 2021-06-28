#include "includes.h"
__global__ void simple_corner_turn_kernel(unsigned short *d_input, float *d_output, int nchans, int nsamp) {

size_t t = blockIdx.x * blockDim.x + threadIdx.x;
size_t c = blockIdx.y * blockDim.y + threadIdx.y;

d_output[(size_t)(c * nsamp) + t] = (float) __ldg(&d_input[(size_t)(t * nchans) + c]);

}