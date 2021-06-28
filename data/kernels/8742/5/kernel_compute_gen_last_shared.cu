#include "includes.h"

#define MAX_CELL_PER_THREAD 1

__global__ void kernel_compute_gen_last_shared(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t rows, uint32_t cols) {
__shared__ int shared[3][128 + 2];

int ix = ((blockDim.x - 2) * blockIdx.x + threadIdx.x) & (cols - 1);
int iy = (blockIdx.y + threadIdx.y) & (rows - 1);
int id = iy * cols + ix;

int i = threadIdx.y;
int j = threadIdx.x;

uint8_t mine = matrix_src[id]; // keep cell in register
shared[i][j] = mine;
//shared[i][j] = matrix_src[id];

__syncthreads();

if (i == 1 && j > 0 && j < 129){

uint8_t aliveCells = shared[i + 1][j] +  // lower
shared[i - 1][j] +  // upper
shared[i][j + 1] +  // right
shared[i][j - 1] +  // left
shared[i + 1][j + 1] +
shared[i - 1][j - 1] +  //diagonals
shared[i - 1][j + 1] +
shared[i + 1][j - 1];

matrix_dst[id] = (aliveCells == 3 || (aliveCells == 2 && mine)) ? 1 : 0;
}
}