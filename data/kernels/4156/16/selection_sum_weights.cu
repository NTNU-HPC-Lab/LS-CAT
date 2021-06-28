#include "includes.h"
__global__ void selection_sum_weights(float * selection_sum,  float * selection, int n, int stride) {
int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;
int idx = 0;
for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
selection_sum[((y+j)*width + x)] = 0;
for ( idx = 0; idx < n; idx ++) {
atomicAdd(&(selection_sum[((y+j)*width + x)]),  selection[idx * stride + ((y+j)*width + x)]);
}
}
}