#include "includes.h"
__global__ void reluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim) {
A[index] = fmaxf(Z[index], 0);
}
}