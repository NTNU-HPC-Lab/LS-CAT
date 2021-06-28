#include "includes.h"
__global__ void is_nan_or_inf_kernel(float *input, size_t size, int *pinned_return)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size) {
float val = input[index];
if (isnan(val) || isinf(val))
*pinned_return = 1;
}
}