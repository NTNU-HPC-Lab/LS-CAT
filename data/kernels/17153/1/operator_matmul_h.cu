#include "includes.h"
__global__ void operator_matmul_h(const float *input1, const float *input2, float *output, int height, int k, int width, int broadcast) {
__shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
__shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

int batch_idx = blockIdx.z;
if (broadcast != 1) input1 += batch_idx * height * k;
if (broadcast != 2) input2 += batch_idx * k * width;
output += batch_idx * height * width;

int bx = blockIdx.y;
int by = blockIdx.x;
int tx = threadIdx.y;
int ty = threadIdx.x;

int row = bx * TILE_SIZE + tx;
int col = by * TILE_SIZE + ty;
float v = 0;

for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
if (i * TILE_SIZE + ty < k && row < height)
shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
else
shared_input1[tx][ty] = 0;

if (i * TILE_SIZE + tx < k && col < width)
shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
else
shared_input2[tx][ty] = 0;
__syncthreads();

for (int j = 0; j < TILE_SIZE; j++)
v += shared_input1[tx][j] * shared_input2[j][ty];
__syncthreads();
}

if (row < height && col < width) output[row * width + col] = v;
}