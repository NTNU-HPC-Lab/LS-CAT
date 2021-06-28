#include "includes.h"

#define N 100000000



__global__ void daxpy_simple(int n, double alpha, double *x, double *y) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
y[idx] += alpha * x[idx];
}
}