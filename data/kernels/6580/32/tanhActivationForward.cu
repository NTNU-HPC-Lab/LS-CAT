#include "includes.h"
__global__ void tanhActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim) {
A[index] = std::tanh(Z[index]);
}
}