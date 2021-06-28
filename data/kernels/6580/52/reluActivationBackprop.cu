#include "includes.h"
__global__ void reluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim) {
if (Z[index] > 0) {
dZ[index] = dA[index];
}
else {
dZ[index] = 0;
}
}
}