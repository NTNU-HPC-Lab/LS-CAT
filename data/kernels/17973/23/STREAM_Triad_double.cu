#include "includes.h"
__global__ void STREAM_Triad_double(double *a, double *b, double *c, double scalar, size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
c[idx] = a[idx]+scalar*b[idx];
idx   += blockDim.x * gridDim.x;
}
}