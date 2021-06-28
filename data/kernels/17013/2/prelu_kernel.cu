#include "includes.h"
__global__ void prelu_kernel(const float *input, float *output, int num_elem, int input_size, int fm_size, const float* gamma) {

int idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx >= num_elem) return;

if (input[idx] >= 0.0f) {
output[idx] = input[idx];
return;
}
int c = (idx % input_size) / fm_size;
output[idx] = input[idx] * gamma[c];
}