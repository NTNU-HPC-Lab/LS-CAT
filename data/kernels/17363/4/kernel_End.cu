#include "includes.h"
__global__ void kernel_End(int *g_stochastic, int *g_count_blocks, int *g_counter)
{
int thid = blockIdx.x * blockDim.x + threadIdx.x;
if (thid < (*g_counter))
{
if (g_stochastic[thid] == 1)
atomicAdd(g_count_blocks, 1);
//(*g_count_blocks) = (*g_count_blocks) + 1 ;
}
}