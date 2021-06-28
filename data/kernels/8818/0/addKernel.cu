#include "includes.h"


__global__ void addKernel(int *c, const int *a, const int *b, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) {
c[i] = a[i] + b[i];
}
}