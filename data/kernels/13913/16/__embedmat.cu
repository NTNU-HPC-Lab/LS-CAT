#include "includes.h"
__global__ void __embedmat(float *a, int *b, long long *c, int n) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
const int signbit = 0x80000000;
const int mag =     0x7fffffff;
for (int i = tid; i < n; i += blockDim.x*gridDim.x*gridDim.y) {
float v = a[i];
int vi = *((int *)&v);
if (vi & signbit) {
vi = -(vi & mag);
}
c[i] = (long long)vi + (((long long)b[i])<<32);
}
}