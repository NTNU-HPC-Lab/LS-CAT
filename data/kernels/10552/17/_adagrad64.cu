#include "includes.h"
__global__ void _adagrad64(int n, double eps, double *dw2, double *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
dw2[i] += dw[i] * dw[i];
dw[i] /= sqrt(dw2[i] + eps);
i += blockDim.x * gridDim.x;
}
}