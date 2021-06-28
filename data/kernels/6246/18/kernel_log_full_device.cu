#include "includes.h"
__global__ void kernel_log_full_device(int *x, int *out, unsigned int size, int epsilon) {
unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < size; i += stride) {
out[i] = (int) log((float) x[i] + epsilon);
}
}