#include "includes.h"
__global__ void transposeFineGrained(float *odata, float *idata, int width, int height)
{
__shared__ float block[TILE_DIM][TILE_DIM+1];

int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
int index = xIndex + (yIndex)*width;

for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
{
block[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
}

__syncthreads();

for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
{
odata[index+i*height] = block[threadIdx.x][threadIdx.y+i];
}
}