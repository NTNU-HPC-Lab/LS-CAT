#include "includes.h"
__global__ void STREAM_Add_double(double *a, double *b, double *c,  size_t len)
{
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < len) {
c[idx] = a[idx]+b[idx];
idx   += blockDim.x * gridDim.x;
}
}