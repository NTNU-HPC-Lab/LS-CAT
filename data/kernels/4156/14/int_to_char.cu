#include "includes.h"
__global__ void int_to_char(int * img2, unsigned char * img)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;
for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
img[3*((y+j)*width + x)] = img2[(y+j)*width + x] / (256*256);
img[3*((y+j)*width + x)+1] = img2[(y+j)*width + x] / 256 % 256;
img[3*((y+j)*width + x)+2] = img2[(y+j)*width + x] % 256;
}
}