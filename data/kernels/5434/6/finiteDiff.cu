#include "includes.h"
__global__ void finiteDiff(const int c, const double dt, const double dx, const int nt, const int nx, double *u, double *un) {

int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int t = 0; t < nt; t++) {

for (int i = index; i < nx; i += stride) {
un[i] = u[i];
}

for (int i = index + 1; i < nx; i += stride) {
u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1]);
}
}
}