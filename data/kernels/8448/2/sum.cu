#include "includes.h"
__global__ void sum(float *a, float *b, float *c) {
int index = blockDim.x * blockIdx.x + threadIdx.x;
if (index < N) {
c[index] = a[index] + b[index];
}
}