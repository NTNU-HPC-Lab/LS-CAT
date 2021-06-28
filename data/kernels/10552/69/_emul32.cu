#include "includes.h"
__global__ void _emul32(int n, float *x, float *y) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
y[i] *= x[i];
i += blockDim.x * gridDim.x;
}
}