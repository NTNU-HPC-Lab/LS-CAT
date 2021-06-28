#include "includes.h"
__global__ void linearLayerUpdateWeights(  float* dZ, float* A, float* W, int dZ_x_dim, int dZ_y_dim, int A_x_dim, int A_y_dim, float learning_rate) {

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// A is treated as transposed
int W_x_dim = A_y_dim;
int W_y_dim = dZ_y_dim;

float dW_value = 0.0f;

if (row < W_y_dim && col < W_x_dim) {
for (int i = 0; i < dZ_x_dim; i++) {
dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
}
W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
}
}