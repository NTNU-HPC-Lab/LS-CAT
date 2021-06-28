#include "includes.h"
__global__ void transform2d(float *points3d_after, float fov_scale)
{
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int w = gridDim.x * TILE_DIM;

for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
{
int iw = x;
int ih = y + j;
float x = points3d_after[(ih * w + iw) * 3 + 0];
float y = points3d_after[(ih * w + iw) * 3 + 1];
float z = points3d_after[(ih * w + iw) * 3 + 2];

points3d_after[(ih * w + iw) * 3 + 0] = x;//sqrt(x * x + y * y + z * z);
//points3d_after[(ih * w + iw) * 3 + 1] = atan2(y, x);
//points3d_after[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);

float x2 = fov_scale * x;
if ((x2 > 0) && (y < x2 * 1.1) && (y > -x2 * 1.1) && (z < x2 * 1.1) && (z > -x2 * 1.1)) {
points3d_after[(ih * w + iw) * 3 + 1] = y / (x2 + 1e-5);
points3d_after[(ih * w + iw) * 3 + 2] = -z / (x2 + 1e-5);
}
else {
points3d_after[(ih * w + iw) * 3 + 1] = -1;
points3d_after[(ih * w + iw) * 3 + 2] = -1;
}
}
}