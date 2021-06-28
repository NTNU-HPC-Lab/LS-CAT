#include "includes.h"
__global__ void _emul64(int n, double *x, double *y) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
y[i] *= x[i];
i += blockDim.x * gridDim.x;
}
}