#include "includes.h"
__global__  void mult_inverse_array_kernel(const float *src_gpu, float *dst_gpu, int size, const float eps)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;

if (index < size) {
float val = src_gpu[index];
float sign = (val < 0) ? -1 : 1;
// eps = 1 by default
// eps = 2 - lower delta
// eps = 0 - higher delta (linear)
// eps = -1 - high delta (inverse number)
dst_gpu[index] = powf(fabs(val), eps) * sign;
}
}