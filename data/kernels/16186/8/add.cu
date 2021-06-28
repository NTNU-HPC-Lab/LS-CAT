#include "includes.h"
__global__ void add(int *a, int *b, int *c ) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
c[index] = a[index] + b[index];
}