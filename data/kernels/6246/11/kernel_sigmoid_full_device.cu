#include "includes.h"
__global__ void kernel_sigmoid_full_device(unsigned int size, int *x, int *out) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < size; i += stride) {
out[i] = 1 / (1 + abs(x[i]));
}
}