#include "includes.h"
__global__ void __extractmat2d(double *a, long long *b, int nrows, int ncols) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
const int signbit = 0x80000000;
const int mag =     0x7fffffff;
for (int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x*gridDim.y) {
int vi = *((int *)&b[i]);
if (vi & signbit) {
vi = -(vi & mag);
}
a[i] = *((double *)&vi);
}
}