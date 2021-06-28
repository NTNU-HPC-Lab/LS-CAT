#include "includes.h"
__global__ void kernelMultMat(double *d_a, double *d_b, double *d_c, int ROWS, int COL_A, int COL_B) {

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
double add;

if (row < ROWS && col < COL_B) {
add = 0;
for (int k = 0; k < COL_A; k++) {
add += d_a[row * COL_A + k] * d_b[k * COL_B + col];
}
d_c[row * COL_B + col] = add;
}
}