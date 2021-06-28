#include "includes.h"
__global__ void _rmsprop32(int n, double eps, double rho, float *dw2, float *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
dw2[i] = dw2[i] * rho + (1 - rho) * dw[i] * dw[i];
dw[i] /= sqrt(dw2[i] + eps);
i += blockDim.x * gridDim.x;
}
}