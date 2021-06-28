#include "includes.h"
/*
* Multiplying a 2D matrix using CUDA
*/


#define BLOCK_SIZE 16


__global__ void gpu_matrix_mul( int *a, int *b, int *c, int m, int n, int k){
int row = blockIdx.y + blockDim.y * threadIdx.y;
int col = blockIdx.x + blockDim.x * threadIdx.x;
int sum = 0;

if(col < k && row < m){
for(int i = 0; i < n; i++){
sum += a[row*n + i] * b[i*k + col];
}
c[row * k + col] = sum;
}
}