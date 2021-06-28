#include "includes.h"
__global__ void mapScan(unsigned int *d_array, unsigned int *d_total, size_t n) {
int tx = threadIdx.x;
int bx = blockIdx.x;
int index = BLOCK_WIDTH * bx + tx;

if(index < n) {
d_array[index] += d_total[bx];
}
}