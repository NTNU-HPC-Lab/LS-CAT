#include "includes.h"
__global__ void transposeKernel(float *inData, float *outData)
{
__shared__ float tile[TILE_DIM][TILE_DIM + 1];

int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;

/* Copying data into shared memory - each thread copies 4 elements : read & write coalesced */
for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
tile[threadIdx.y + j][threadIdx.x] = inData[(y+j) * width + x];

__syncthreads();

/* x,y modified according to the new transposed matrix */
x = blockIdx.y * TILE_DIM + threadIdx.x;
y = blockIdx.x * TILE_DIM + threadIdx.y;

/* Copying data to output array - each thread copies 4 elemets : read & write coalesced */
for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
outData[(y+j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}