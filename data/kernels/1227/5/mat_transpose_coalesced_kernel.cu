#include "includes.h"
__global__ void mat_transpose_coalesced_kernel(int *mat, int *res) {
// Square tile
int tile_dim = 32;
// 32 Blocks across for 1024 mat
int blocks_per_row = 32;

__shared__ int smem[32 * 32];

int rows_per_block_iter = 64;
// Each iter has 2 "block-rows"
for (int block_iter = 0; block_iter < 16; block_iter++) {
int tile_row = blockIdx.x / blocks_per_row;
int tile_col = blockIdx.x % blocks_per_row;

int intile_row = threadIdx.x / tile_dim;
int intile_col = threadIdx.x % tile_dim;

int read_row = (tile_row * tile_dim) + intile_row + (rows_per_block_iter * block_iter);
int read_col = (tile_col * tile_dim) + intile_col;

int write_row = (tile_col * tile_dim) + intile_row;
int write_col = (tile_row * tile_dim) + intile_col + (rows_per_block_iter * block_iter);


smem[(intile_row * tile_dim) + intile_col] = mat[(read_row * 1024) + read_col];
__syncthreads();
res[(write_row * 1024) + write_col] = smem[(intile_col * tile_dim) + intile_row];
}
}