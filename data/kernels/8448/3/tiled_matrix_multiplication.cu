#include "includes.h"
__global__ void tiled_matrix_multiplication(int *A, int *B, int *C) {

__shared__ int As[TILE_WIDTH][TILE_WIDTH];
__shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

int res = 0;

for(int i = 0; i < M/TILE_WIDTH; i++) {
As[ty][tx] = A[row * M + (i*TILE_WIDTH + tx)];
Bs[ty][tx] = B[(i*TILE_WIDTH + ty)* M + col];

__syncthreads();

for(int j = 0; j < TILE_WIDTH; j++) {
res += As[ty][j] + Bs[j][tx];
}

__syncthreads();
}

C[row * M + col] = res;

}