#include "includes.h"
__global__ void to3d_point(float *depth, float *points3d)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int w = gridDim.x * TILE_DIM;
int h = w / 2;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{

int iw = x;
int ih = y + j;
float depth_point = depth[ ih*w + iw ] * 128.0;
float phi = ((float)(ih) + 0.5) / float(h) * M_PI;
float theta = ((float)(iw) + 0.5) / float(w) * 2 * M_PI + M_PI;

points3d[(ih * w + iw) * 4 + 0] = depth_point * sin(phi) * cos(theta);
points3d[(ih * w + iw) * 4 + 1] = depth_point * sin(phi) * sin(theta);
points3d[(ih * w + iw) * 4 + 2] = depth_point * cos(phi);
points3d[(ih * w + iw) * 4 + 3] = 1;

}
}