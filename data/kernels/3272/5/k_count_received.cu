#include "includes.h"
__global__ static void k_count_received(int nr_total_blocks, uint* d_n_recv_by_block, uint* d_spine_cnts)
{
int bid = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

if (bid < nr_total_blocks) {
d_spine_cnts[bid * 10 + CUDA_BND_S_NEW] = d_n_recv_by_block[bid];
}
}