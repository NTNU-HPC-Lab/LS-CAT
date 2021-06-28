#include "includes.h"
__device__ float sigmoid(float x) {
return 1.0f / (1 + __expf(-x));
}
__global__ void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim) {
A[index] = sigmoid(Z[index]);
}
}