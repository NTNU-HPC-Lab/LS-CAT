#include "includes.h"
__global__ void __embedmat2d(float *a, long long *b, int nrows, int ncols, int sortdown) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
const int signbit = 0x80000000;
const int mag =     0x7fffffff;
int icol = 0;
for (int i = tid; i < nrows*ncols; i += blockDim.x*gridDim.x*gridDim.y) {
float v = a[i];
int vi = *((int *)&v);
if (vi & signbit) {
vi = -(vi & mag);
}
icol = i/nrows+1;
if (sortdown) icol = ncols - icol + 1;
b[i] = (long long)vi + (((long long)icol)<<32);
}
}