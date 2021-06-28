#include "includes.h"
__global__ void matrixMul(int *a, int *b, int *c, int n, int tile_size){
__shared__ int A[SHMEM_SIZE];
__shared__ int B[SHMEM_SIZE];

int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;

int row = by * tile_size + ty;
int col = bx * tile_size + tx;

int temp_sum = 0;

for (int i = 0; i < (n / tile_size); i++){
A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

__syncthreads();

for(int j = 0; j < tile_size; j++){
temp_sum += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
}

__syncthreads();
}

c[(row * n) + col] = temp_sum;
}