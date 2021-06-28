#include "includes.h"
__global__ void multi(float *a, float *b, float *c, int width) {
__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float result = 0;

for (int p = 0; p < width/TILE_WIDTH; p++)
{
s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*TILE_WIDTH + threadIdx.x)];
s_b[threadIdx.y][threadIdx.x] = b[(p*TILE_WIDTH + threadIdx.y)*width + col];

__syncthreads();

for (int i = 0; i < TILE_WIDTH; i++)
{
result += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
}

__syncthreads();
}

c[row * width + col] = result;
}