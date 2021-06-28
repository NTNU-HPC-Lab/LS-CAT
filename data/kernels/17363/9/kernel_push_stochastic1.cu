#include "includes.h"
__global__ void kernel_push_stochastic1(int *g_push_reser, int *s_push_reser, int *g_count_blocks, bool *g_finish, int *g_block_num, int width1)
{
int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
int thid = __umul24(y, width1) + x;

s_push_reser[thid] = g_push_reser[thid];

if (thid == 0)
{
if ((*g_count_blocks) == 0)
(*g_finish) = false;
}
}