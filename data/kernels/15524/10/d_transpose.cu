#include "includes.h"

__constant__ float *c_Kernel;

__global__ void d_transpose(float *odata, float *idata, int width, int height)
{
__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

// read the matrix tile into shared memory
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

if ((xIndex < width) && (yIndex < height))
{
unsigned int index_in = yIndex * width + xIndex;
block[threadIdx.y][threadIdx.x] = idata[index_in];
}

__syncthreads();

// write the transposed matrix tile to global memory
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

if ((xIndex < height) && (yIndex < width))
{
unsigned int index_out = yIndex * height + xIndex;
odata[index_out] = block[threadIdx.x][threadIdx.y];
}
}