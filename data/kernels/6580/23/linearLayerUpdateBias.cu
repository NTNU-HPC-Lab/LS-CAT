#include "includes.h"
__global__ void linearLayerUpdateBias(float* dZ, float* b, int dZ_x_dim, int dZ_y_dim, int b_x_dim, float learning_rate) {
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < dZ_x_dim * dZ_y_dim) {
int dZ_x = index % dZ_x_dim;
int dZ_y = index / dZ_x_dim;
atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
}
}