#include "includes.h"
__global__ void set_array_double(double *a,  double value, size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
a[idx] = value;
idx   += blockDim.x * gridDim.x;
}
}