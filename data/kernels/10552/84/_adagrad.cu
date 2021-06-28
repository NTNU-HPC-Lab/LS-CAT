#include "includes.h"
__global__ void _adagrad(int n, float eps, float *dw2, float *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
dw2[i] += dw[i] * dw[i];
dw[i] /= (eps + sqrt(dw2[i]));
i += blockDim.x * gridDim.x;
}
}