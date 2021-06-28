#include "includes.h"
__global__ void __extractmat(double *a, int *b, long long *c, int n) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
const int signbit = 0x80000000;
const int mag =     0x7fffffff;
for (int i = tid; i < n; i += blockDim.x*gridDim.x*gridDim.y) {
int vi = *((int *)&c[i]);
if (vi & signbit) {
vi = -(vi & mag);
}
a[i] = *((double *)&vi);
b[i] = *(((int *)&c[i])+1);
}
}