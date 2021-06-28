#include "includes.h"
__global__ void mapPredicate(unsigned int *d_zeros, unsigned int *d_ones, unsigned int *d_in, unsigned int bit, size_t n)
{
int tx = threadIdx.x;
int bx = blockIdx.x;
int index = BLOCK_WIDTH * bx + tx;

if(index < n) {
unsigned int isOne = (d_in[index] >> bit) & 1;
d_ones[index] = isOne;
d_zeros[index] = 1 - isOne;
}
}