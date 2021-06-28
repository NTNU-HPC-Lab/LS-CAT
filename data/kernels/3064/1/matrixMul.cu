#include "includes.h"
__global__ void matrixMul(int *a, int *b, int *c, int n){
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

int temp_sum = 0;

if((row < n) && (col < n)){
for (int k = 0; k < n; k++){
temp_sum += a[row * n + k] * b[k * n + col];
}

c[row * n + col] = temp_sum;
}

}