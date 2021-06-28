#include "includes.h"
__global__ void scan_v1_kernel(float *d_output, float *d_input, int length)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x;

float element = 0.f;
for (int offset = 0; offset < length; offset++) {
if (idx - offset >= 0)
element += d_input[idx - offset];
}
d_output[idx] = element;
}