#include "includes.h"
__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{
__shared__ float tile[TILE_DIM][TILE_DIM+1];

int blockIdx_x, blockIdx_y;

// do diagonal reordering
if (width == height)
{
blockIdx_y = blockIdx.x;
blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
}
else
{
int bid = blockIdx.x + gridDim.x*blockIdx.y;
blockIdx_y = bid%gridDim.y;
blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
}

// from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
// and similarly for y

int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
int index_in = xIndex + (yIndex)*width;

xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
int index_out = xIndex + (yIndex)*height;

for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
{
tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
}

__syncthreads();

for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
{
odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
}
}