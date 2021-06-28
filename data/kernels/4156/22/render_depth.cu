#include "includes.h"
__global__ void render_depth(float *points3d_polar, unsigned int * depth_render)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int w = gridDim.x * TILE_DIM;
int h = w /2;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{
int iw = x;
int ih = y + j;
int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
int ty = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h - 0.5);
int this_depth = (int)(512 * points3d_polar[(ih * w + iw) * 3 + 0]);
atomicMin(&depth_render[(ty * w + tx)] , this_depth);
}
}