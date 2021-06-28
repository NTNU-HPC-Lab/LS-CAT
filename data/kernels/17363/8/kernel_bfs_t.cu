#include "includes.h"
__global__ void kernel_bfs_t(int *g_push_reser, int  *g_sink_weight, int *g_graph_height, bool *g_pixel_mask, int vertex_num, int width, int height, int vertex_num1, int width1, int height1)
{

int thid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

if (thid < vertex_num && g_pixel_mask[thid] == true)
{
int col = thid % width1, row = thid / width1;

if (col > 0 && row > 0 && col < width - 1 && row < height - 1 && g_push_reser[thid] > 0)
{
g_graph_height[thid] = 1;
g_pixel_mask[thid] = false;
}
else
if (g_sink_weight[thid] > 0)
{
g_graph_height[thid] = -1;
g_pixel_mask[thid] = false;
}
}
}