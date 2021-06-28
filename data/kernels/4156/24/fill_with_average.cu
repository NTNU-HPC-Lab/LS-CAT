#include "includes.h"
__global__ void fill_with_average(unsigned char *img, int * nz, int * average, int scale)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;
//int h = width /2;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{
int iw = x;
int ih = y + j;

if ((img[3*(ih*width + iw)] + img[3*(ih*width + iw)+1] + img[3*(ih*width + iw)+2] == 0) && (nz[ih/scale * width + iw/scale] > 0))
{
img[3*(ih*width + iw)] = (unsigned char)(average[3*(ih/scale*width + iw/scale)] / nz[ih/scale * width + iw/scale]);
img[3*(ih*width + iw) + 1] = (unsigned char)(average[3*(ih/scale*width + iw/scale) + 1] / nz[ih/scale * width + iw/scale]);
img[3*(ih*width + iw) + 2] = (unsigned char)(average[3*(ih/scale*width + iw/scale) + 2] / nz[ih/scale * width + iw/scale]);
}

}
}