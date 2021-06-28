#include "includes.h"
__global__ void vecProduct(int *d_x, int *d_y, int *d_z, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
d_z[idx] = d_x[idx] * d_y[idx];
}
}