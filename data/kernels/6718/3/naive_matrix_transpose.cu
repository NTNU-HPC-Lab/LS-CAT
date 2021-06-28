#include "includes.h"
__global__ void naive_matrix_transpose(float *input, int axis_0, int axis_1, float *output)
{
__shared__ float tile[TILE_DIM][TILE_DIM + 1];

int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
for (int i = 0; i < TILE_DIM && y + i < axis_1 && x < axis_0; i += BLOCK_HEIGHT) {
tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * axis_0 + x];
}

__syncthreads();

x = blockIdx.y * TILE_DIM + threadIdx.x;
y = blockIdx.x * TILE_DIM + threadIdx.y;

for (int i = 0; i < TILE_DIM && y + i < axis_1 && x < axis_0; i += BLOCK_HEIGHT) {
output[(y + i) * axis_0 + x] = tile[(threadIdx.x)][threadIdx.y + i];
}
}