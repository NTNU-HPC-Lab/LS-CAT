#include "includes.h"
__global__ static void mprts_update_offsets(int nr_total_blocks, uint *d_off, uint *d_spine_sums)
{
int bid = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

if (bid <= nr_total_blocks) {
d_off[bid] = d_spine_sums[bid * CUDA_BND_STRIDE + 0];
}
}