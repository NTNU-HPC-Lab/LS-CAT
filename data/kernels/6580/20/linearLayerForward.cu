#include "includes.h"
__global__ void linearLayerForward( float* W, float* A, float* Z, float* b, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim) {

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

int Z_x_dim = A_x_dim;
int Z_y_dim = W_y_dim;

float Z_value = 0;

if (row < Z_y_dim && col < Z_x_dim) {
for (int i = 0; i < W_x_dim; i++) {
Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
}
Z[row * Z_x_dim + col] = Z_value + b[row];
}
}