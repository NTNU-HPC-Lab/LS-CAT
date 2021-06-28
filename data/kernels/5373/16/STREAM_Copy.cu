#include "includes.h"
__global__ void STREAM_Copy(float *a, float *b, size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
b[idx] = a[idx];
idx   += blockDim.x * gridDim.x;
}
}