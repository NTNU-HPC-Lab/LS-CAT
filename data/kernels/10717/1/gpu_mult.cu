#include "includes.h"
__global__ void gpu_mult(int *a,int *b, int *c, int N) {

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int sum = 0;
if( col < N && row < N) {
for(int i = 0; i < N; i++) {
sum += a[row * N + i] * b[i * N + col];
}
c[row * N + col] = sum;
}
}