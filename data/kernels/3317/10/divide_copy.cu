#include "includes.h"
__global__ void divide_copy(double *dest, const double *src, int length, const double divisor)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
double factor = 1.0 / divisor;
while (tid < length) {
dest[tid] = src[tid] * factor;
tid += blockDim.x * gridDim.x;
}
}