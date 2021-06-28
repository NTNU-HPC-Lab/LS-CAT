#include "includes.h"
__global__ void addBias(float* Z, float* b, int Z_x_dim, int Z_y_dim){
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if(row < Z_y_dim && col < Z_x_dim){
Z[row * Z_x_dim + col] += b[row];
}
}