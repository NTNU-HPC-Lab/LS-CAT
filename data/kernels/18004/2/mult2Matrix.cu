#include "includes.h"
__global__ void mult2Matrix(float *M, float *N, float *P) {
__shared__ int shared_m_tile[TILE_WIDTH][TILE_WIDTH];
__shared__ int shared_n_tile[TILE_WIDTH][TILE_WIDTH];

int tx = threadIdx.x;
int ty = threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
//check if thread directly maps to the dimensions of the resulting matrix
if (row < WIDTH && col < WIDTH)
{
float result = 0;
int k;
int phase;
//calculate P matrix indexes in phases. Each phase shares
//TILE_SIZE * TILE_SIZE data copied to the shared matrix M
//and matrix N.
for (phase = 0; phase <= WIDTH / TILE_WIDTH; phase++)
{
shared_m_tile[ty][tx] = M[row * WIDTH + phase * TILE_WIDTH + tx];
shared_n_tile[ty][tx] = N[(phase * TILE_WIDTH + ty) * WIDTH + col];
__syncthreads();

for (k = 0; k < TILE_WIDTH; k++)
{
if (k + (phase * TILE_WIDTH) < WIDTH)
{
result += (shared_m_tile[ty][k] * shared_n_tile[k][tx]);
}
}
__syncthreads();
}
P[row * WIDTH + col] = result;
}
}