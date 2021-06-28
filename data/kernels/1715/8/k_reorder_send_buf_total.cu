#include "includes.h"
__global__ static void k_reorder_send_buf_total(int nr_prts, int nr_total_blocks, uint *d_bidx, uint *d_sums, float4 *d_xi4, float4 *d_pxi4, float4 *d_xchg_xi4, float4 *d_xchg_pxi4)
{
int i = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
if (i >= nr_prts)
return;

if (d_bidx[i] == CUDA_BND_S_OOB) {
int j = d_sums[i];
d_xchg_xi4[j]  = d_xi4[i];
d_xchg_pxi4[j] = d_pxi4[i];
}
}