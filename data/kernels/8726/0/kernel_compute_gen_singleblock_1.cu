#include "includes.h"

#define MAX_CELL_PER_THREAD 1

__device__ void gpu_swap(void **a, void **b) {
void *temp = *a;
*a = *b;
*b = temp;
}
__global__ void kernel_compute_gen_singleblock_1(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols, uint32_t iterations) {
uint32_t cell = threadIdx.x;

uint32_t x1 = cell     & cols-1; //% cols;
uint32_t x0 = (x1 - 1) & cols-1; //% cols;
uint32_t x2 = (x1 + 1) & cols-1; //% cols;

uint32_t y1 = cell - x1;
uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
uint32_t y2 = (y1 + cols) & dim_m1; //% dim;

for (uint32_t iter = 0; iter < iterations; iter++){
uint8_t aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0;
gpu_swap((void**)&matrix_dst,(void**)&matrix_src);
__syncthreads();
}
}