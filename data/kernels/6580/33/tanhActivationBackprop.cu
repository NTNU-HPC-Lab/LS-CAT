#include "includes.h"
__global__ void tanhActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim) {
float d = Z[index];
dZ[index] = dA[index] * (1 - d * d);
}
}