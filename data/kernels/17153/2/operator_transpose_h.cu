#include "includes.h"
__global__ void operator_transpose_h(const float *in, float *out, int height, int width) {
__shared__ float tile[TILE_SIZE][TILE_SIZE];

int batch_idx = blockIdx.z;
in += batch_idx * height * width;
out += batch_idx * height * width;

int bx = blockIdx.y;
int by = blockIdx.x;
int tx = threadIdx.y;
int ty = threadIdx.x;

int row = bx * TILE_SIZE + tx;
int col = by * TILE_SIZE + ty;

if (row < height && col < width) {
// coalesced read from global mem, TRANSPOSED write into shared mem:
tile[tx][ty] = in[row * width + col];

__syncthreads();

// read from shared mem, coalesced write to global mem
out[col * height + row] = tile[tx][ty];
}
}