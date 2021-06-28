#include "includes.h"
__global__ void kernel_pow_grad_device(int *x, int power, int *grad, int *out, bool grad_is_scalar, unsigned int size) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < size; i += stride) {
out[i] = grad[(grad_is_scalar) ? 0 : i] * ((int) power) * ((int) powf((float) x[i], power - 1));
}
}