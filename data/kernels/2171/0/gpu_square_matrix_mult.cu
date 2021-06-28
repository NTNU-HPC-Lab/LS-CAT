#include "includes.h"
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n)
{
__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
int tmp = 0;
int idx;

for (int sub = 0; sub < gridDim.x; ++sub)
{
idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
if(idx >= n*n)
{
// n may not divisible by BLOCK_SIZE
tile_a[threadIdx.y][threadIdx.x] = 0;
}
else
{
tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
}

idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
if(idx >= n*n)
{
tile_b[threadIdx.y][threadIdx.x] = 0;
}
else
{
tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
}
__syncthreads();

for (int k = 0; k < BLOCK_SIZE; ++k)
{
tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
}
__syncthreads();
}
if(row < n && col < n)
{
d_result[row * n + col] = tmp;
}
}