#include "includes.h"
__device__ float sigmoid(float x) {
return 1.0f / (1 + __expf(-x));
}
__global__ void transpose_sigmoid(float *odata, float *idata, int width, int height)
{
__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

// read the matrix tile into shared memory
// load one element per thread from device memory (idata) and store it
// in transpose_relud order in block[][]
unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
if((xIndex < width) && (yIndex < height))
{
unsigned int index_in = yIndex * width + xIndex;
block[threadIdx.y][threadIdx.x] = idata[index_in];
}

// synchronise to ensure all writes to block[][] have completed
__syncthreads();

// write the transpose_relud matrix tile to global memory (odata) in linear order
xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
if((xIndex < height) && (yIndex < width))
{
unsigned int index_out = yIndex * height + xIndex;
odata[index_out] = block[threadIdx.x][threadIdx.y];
}
}