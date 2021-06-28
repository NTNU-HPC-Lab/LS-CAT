#include "includes.h"
__global__ void sum(int *a, int *b, int *c) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
while (i < N) {
c[i] = a[i] + b[i];
i += gridDim.x * blockDim.x;
}
}