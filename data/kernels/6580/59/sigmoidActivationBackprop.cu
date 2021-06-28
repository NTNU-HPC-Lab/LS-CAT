#include "includes.h"
__device__ float sigmoid(float x) {
return 1.0f / (1 + __expf(-x));
}
__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < Z_x_dim * Z_y_dim){
dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
}
}