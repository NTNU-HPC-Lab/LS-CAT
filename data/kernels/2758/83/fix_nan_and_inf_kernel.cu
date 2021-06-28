#include "includes.h"
__global__ void fix_nan_and_inf_kernel(float *input, size_t size)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size) {
float val = input[index];
if (isnan(val) || isinf(val))
input[index] = 1.0f / index;  // pseudo random value
}
}