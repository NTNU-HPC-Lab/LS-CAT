#include "includes.h"


__global__ void cpy(float *a, float *b, int n) {
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n)
a[i] = b[i];
}