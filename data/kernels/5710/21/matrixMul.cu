#include "includes.h"
__global__ void matrixMul(float* A, float* B, float* C, int width)
{
__shared__ float As[TILE_WIDTH] [TILE_WIDTH];
__shared__ float Bs[TILE_WIDTH] [TILE_WIDTH];
int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
float c_val = 0.0f;for(int i = 0; i < width/TILE_WIDTH; i++)
{
As[threadIdx.y][threadIdx.x] = A[row * width + (i * TILE_WIDTH + threadIdx.x)];
Bs[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * width + col ];
__syncthreads();
for(int k = 0; k < TILE_WIDTH; k++)
c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];__syncthreads();
}
C[row * width + col] = c_val;
}