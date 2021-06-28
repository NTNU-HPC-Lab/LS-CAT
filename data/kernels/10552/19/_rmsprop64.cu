#include "includes.h"
__global__ void _rmsprop64(int n, double eps, double rho, double *dw2, double *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
dw2[i] = dw2[i] * rho + (1 - rho) * dw[i] * dw[i];
dw[i] /= sqrt(dw2[i] + eps);
i += blockDim.x * gridDim.x;
}
}