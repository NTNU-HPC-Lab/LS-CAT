#include "includes.h"

#define N 100000000



__global__ void daxpy(int n, double alpha, double *x, double *y) {
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
idx < n;
idx += blockDim.x * gridDim.x) {
y[idx] += alpha * x[idx];
}
}