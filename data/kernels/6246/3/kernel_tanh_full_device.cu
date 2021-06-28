#include "includes.h"
__global__ void kernel_tanh_full_device(unsigned int size, int *x, int *out) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

/* tanh : R -> (-1,1)  which is 0 in the integers */
for (unsigned int i = idx; i < size; i += stride) {
out[i] = 0;
}
}