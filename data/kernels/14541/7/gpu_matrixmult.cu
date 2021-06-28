#include "includes.h"
__global__ void gpu_matrixmult(int *gpu_a, int *gpu_b, int *gpu_c, int N) {

int k, sum = 0;
int col = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;

if(col < N && row < N) {
for(k = 0; k < N; k++)
sum += gpu_a[row * N + k] * gpu_b[k * N + col];
gpu_c[row * N + col] = sum;
}
}