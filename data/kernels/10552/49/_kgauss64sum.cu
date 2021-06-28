#include "includes.h"
__global__ void _kgauss64sum(int xrows, int xcols, double *x, double *xx) {
int i, j, x0, x1;
double sum;
j = threadIdx.x + blockIdx.x * blockDim.x;
while (j < xcols) {
x0 = j*xrows; x1 = x0+xrows;
sum = 0;
for (i=x0; i<x1; i++) sum += x[i]*x[i];
xx[j] = sum;
j += blockDim.x * gridDim.x;
}
}