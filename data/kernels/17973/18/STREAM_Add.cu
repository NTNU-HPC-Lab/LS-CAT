#include "includes.h"
__global__ void STREAM_Add(float *a, float *b, float *c,  size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
c[idx] = a[idx]+b[idx];
idx   += blockDim.x * gridDim.x;
}
}