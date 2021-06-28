#include "includes.h"
__global__ void swap(unsigned short *d_input, float *d_output, int nchans, int nsamp) {

size_t t = blockIdx.x * blockDim.x + threadIdx.x;
size_t c = blockIdx.y * blockDim.y + threadIdx.y;

d_input[(size_t)(c * nsamp) + t] = (unsigned short) __ldg(&d_output[(size_t)(c * nsamp) + t]);

}