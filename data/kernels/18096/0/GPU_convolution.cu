#include "includes.h"

#define TILE_WIDTH 32

struct event_pair
{
cudaEvent_t start;
cudaEvent_t end;
};

__global__ void GPU_convolution(float *channel, float *mask, float *result, int dimMask, int dimW, int dimH) {
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int x, y;

// Identify the row and column of the Pd element to work on
int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;

int nidRow = Row - dimMask / 2;
int nidCol = Col - dimMask / 2;

int tid = Row * dimW + Col;

if (tid < dimW * dimH) {
result[tid] = 0;
for (int i = 0; i < dimMask; ++i) {
x = nidRow * dimW + i * dimW;
for (int j = 0; j < dimMask; ++j) {
y = nidCol + j;
// When the value is not beyond the borders
if (x >= 0 && y >= 0 && x < dimW * dimH && y < dimW) {
result[tid] += mask[dimMask * i + j] * channel[x + y];
}
}
}
if (result[tid] > 255)
result[tid] = 255;
if (result[tid] < 0)
result[tid] = 0;
}
}