#include "includes.h"
__global__ void STREAM_Triad(float *a, float *b, float *c, float scalar, size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
c[idx] = a[idx]+scalar*b[idx];
idx   += blockDim.x * gridDim.x;
}
}