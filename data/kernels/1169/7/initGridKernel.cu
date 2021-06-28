#include "includes.h"
__global__ void initGridKernel ( float *d_grid, int axis, int w, int h, int d ) {
const int baseX = blockIdx.x * IG_BLOCKDIM_X + threadIdx.x;
const int baseY = blockIdx.y * IG_BLOCKDIM_Y + threadIdx.y;
const int baseZ = blockIdx.z * IG_BLOCKDIM_Z + threadIdx.z;

const int idx = (baseZ * h + baseY) * w + baseX;

if(axis == 0) {
d_grid[idx] = (float)baseX;
} else if(axis == 1) {
d_grid[idx] = (float)baseY;
} else {
d_grid[idx] = (float)baseZ;
}

}