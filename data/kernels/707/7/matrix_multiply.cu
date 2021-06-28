#include "includes.h"
__global__ void matrix_multiply(float *a, float *b, float *c, int num, size_t width)
{
// create shorthand names for threadIdx & blockIdx
int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x, by = blockIdx.y;

// allocate 2D tiles in __shared__ memory
__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

// calculate the row & column index of the element
int row = by * blockDim.y + ty;
int col = bx * blockDim.x + tx;

float result = 0;

// loop over the tiles of the input in phases
for(int i = 0; i < (width - 1)/TILE_WIDTH + 1; ++i)
{
// collaboratively load tiles into __shared__
if (row < width && i*TILE_WIDTH + tx < width)
{
s_a[ty][tx] = a[row*width + i*TILE_WIDTH + tx];
}
else
{
s_a[ty][tx] = 0.0;
}
if (col < width && i*TILE_WIDTH + ty < width)
{
s_b[ty][tx] = b[(i*TILE_WIDTH + ty)*width + col];
}
else
s_b[ty][tx] = 0.0;


// wait until all data is loaded before allowing any thread in this block to continue
__syncthreads();

// do dot product between row of s_a and column of s_b
for(int k = 0; k < TILE_WIDTH; ++k)
{
result += s_a[ty][k] * s_b[k][tx];
}

// wait until all threads are finished with the data before allowing any thread in this block to continue
__syncthreads();
}

if (row < width && col < width)
{
c[row*num + col] = result;
}
}