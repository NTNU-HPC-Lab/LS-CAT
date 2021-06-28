#include "includes.h"
__global__ void VecAdd(int *a, int *b, int *c, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < n) {
c[i] = a[i] + b[i];
}
}