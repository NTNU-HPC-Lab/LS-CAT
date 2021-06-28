#include "includes.h"
__global__ void kTranspose(float *odata, float *idata, int width, int height) {
__shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

// read the matrix tile into shared memory
unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

if((xIndex < width) && (yIndex < height)) {
unsigned int index_in = yIndex * width + xIndex;

block[threadIdx.y][threadIdx.x] = idata[index_in];
}

__syncthreads();

// write the transposed matrix tile to global memory
xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

if((xIndex < height) && (yIndex < width)) {
unsigned int index_out = yIndex * height + xIndex;

odata[index_out] = block[threadIdx.x][threadIdx.y];
}
}