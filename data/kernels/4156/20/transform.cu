#include "includes.h"
__global__ void transform(float *points3d_after, float *points3d, float * transformation_matrix)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int w = gridDim.x * TILE_DIM;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{
int iw = x;
int ih = y + j;
for (int ic = 0; ic < 3; ic ++) {
points3d_after[(ih * w + iw) * 3 + ic] = points3d[(ih * w + iw) * 4 + 0] * transformation_matrix[4 * ic + 0]
+ points3d[(ih * w + iw) * 4 + 1] * transformation_matrix[4 * ic + 1]
+ points3d[(ih * w + iw) * 4 + 2] * transformation_matrix[4 * ic + 2]
+ points3d[(ih * w + iw) * 4 + 3] * transformation_matrix[4 * ic + 3];
}
}
}