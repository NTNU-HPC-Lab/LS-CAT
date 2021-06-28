#include "includes.h"
__global__ void copy_mem(unsigned char *source, unsigned char *render)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
for (int channel = 0; channel < 3; channel ++ )
render[3*((y+j)*width + x) + channel] = source[3 * ((y+j)*width + x) + channel];
}