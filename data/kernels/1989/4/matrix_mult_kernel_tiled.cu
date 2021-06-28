#include "includes.h"
__global__ void matrix_mult_kernel_tiled(int *d_m, int *d_n, int *d_p, int m, int n, int k) {
/*
* [m][k] @ [k][n] = [m][n]
*/
__shared__ int ds_m[TILE_WIDTH][TILE_WIDTH]; // ds: device shared memory
__shared__ int ds_n[TILE_WIDTH][TILE_WIDTH];

int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

int pvalue = 0;

for (int i = 0; i < ceil(k / (float)TILE_WIDTH); ++i) {
// thread collaborative loading into shared memory
if (row < m && (i * TILE_WIDTH + tx) < k)
ds_m[ty][tx] = d_m[row * k + i * TILE_WIDTH + tx]; // coalesced
else
ds_m[ty][tx] = 0;
if (col < n && (i * TILE_WIDTH + ty) < k)
ds_n[ty][tx] = d_n[(i * TILE_WIDTH + ty) * n + col]; // coalesced
else
ds_n[ty][tx] = 0;

__syncthreads();

for (int j = 0; j < TILE_WIDTH; j++)
pvalue += ds_m[ty][j] * ds_n[j][tx];
__syncthreads();
}

if (row < m && col < n)
d_p[row * n + col] = pvalue;
}