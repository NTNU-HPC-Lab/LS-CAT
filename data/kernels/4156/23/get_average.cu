#include "includes.h"
__global__ void get_average(unsigned char * img, int * nz, int * average, int scale)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;
//int h = width /2;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{
int iw = x;
int ih = y + j;

if (img[3*(ih*width + iw)] + img[3*(ih*width + iw)+1] + img[3*(ih*width + iw)+2] > 0)
{
//nz[ih/3 * width + iw/3] += 1;
//average[3*(ih/3*width + iw/3)] += (int)img[3*(ih*width + iw)];
//average[3*(ih/3*width + iw/3)+1] += (int)img[3*(ih*width + iw)+1];
//average[3*(ih/3*width + iw/3)+2] += (int)img[3*(ih*width + iw)+2];

atomicAdd(&(nz[ih/scale * width + iw/scale]), 1);
atomicAdd(&(average[3*(ih/scale*width + iw/scale)]), (int)img[3*(ih*width + iw)]);
atomicAdd(&(average[3*(ih/scale*width + iw/scale)+1]), (int)img[3*(ih*width + iw)+1]);
atomicAdd(&(average[3*(ih/scale*width + iw/scale)+2]), (int)img[3*(ih*width + iw)+2]);

}

}
}