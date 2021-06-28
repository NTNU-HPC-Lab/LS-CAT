#include "includes.h"
__global__ void STREAM_Scale(float *a, float *b, float scale,  size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
b[idx] = scale* a[idx];
idx   += blockDim.x * gridDim.x;
}
}