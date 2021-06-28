#include "includes.h"
__global__ void linearLayerBackprop(float* W, float* dZ, float *dA, int W_x_dim, int W_y_dim, int dZ_x_dim, int dZ_y_dim) {

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// W is treated as transposed
int dA_x_dim = dZ_x_dim;
int dA_y_dim = W_x_dim;

float dA_value = 0.0f;

if (row < dA_y_dim && col < dA_x_dim) {
for (int i = 0; i < W_y_dim; i++) {
dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
}
dA[row * dA_x_dim + col] = dA_value;
}
}