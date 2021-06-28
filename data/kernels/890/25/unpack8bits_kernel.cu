#include "includes.h"
__global__ void unpack8bits_kernel(float *rcp, float *lcp, const int8_t *src) {

const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
const size_t j = i*2;

rcp[i] = static_cast<float>(src[j]);
lcp[i] = static_cast<float>(src[j+1]);
}