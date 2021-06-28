#include "includes.h"
__global__ void gpu_multiply(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

float CValue = 0;

int Row = blockIdx.y*TILE_DIM + threadIdx.y;
int Col = blockIdx.x*TILE_DIM + threadIdx.x;

__shared__ float As[TILE_DIM][TILE_DIM];
__shared__ float Bs[TILE_DIM][TILE_DIM];

for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
As[threadIdx.y][threadIdx.x] = A[Row*ACols + k * TILE_DIM + threadIdx.x];
else
As[threadIdx.y][threadIdx.x] = 0.0;

if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
else
Bs[threadIdx.y][threadIdx.x] = 0.0;
__syncthreads();

for (int n = 0; n < TILE_DIM; ++n)
CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

__syncthreads();
}

if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;
}