#include "includes.h"
__global__ void copySharedMem(float *odata, float *idata, int width, int height)
{
__shared__ float tile[TILE_DIM][TILE_DIM];

int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

int index  = xIndex + width*yIndex;

for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
{
if (xIndex < width && yIndex < height)
{
tile[threadIdx.y][threadIdx.x] = idata[index];
}
}

__syncthreads();

for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
{
if (xIndex < height && yIndex < width)
{
odata[index] = tile[threadIdx.y][threadIdx.x];
}
}
}