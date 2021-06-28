#include "includes.h"
__global__ void kernel_bfs(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight, int *g_graph_height, bool *g_pixel_mask, int vertex_num, int width, int height, int vertex_num1, int width1, int height1, bool *g_over, int *g_counter)
{
/*******************************
*threadId is calculated ******
*****************************/

int thid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

if (thid < vertex_num && g_pixel_mask[thid] == true)
{
int col = thid % width1, row = thid / width1;

if (col < width - 1 && col > 0 && row < height - 1 && row > 0)
{
int height_l = 0, height_d = 0, height_u = 0, height_r = 0;
height_r = g_graph_height[thid + 1];
height_l = g_graph_height[thid - 1];
height_d = g_graph_height[thid + width1];
height_u = g_graph_height[thid - width1];

if (((height_l == (*g_counter) && g_right_weight[thid - 1] > 0)) || ((height_d == (*g_counter) && g_up_weight[thid + width1] > 0) || (height_r == (*g_counter) && g_left_weight[thid + 1] > 0) || (height_u == (*g_counter) && g_down_weight[thid - width1] > 0)))
{
g_graph_height[thid] = (*g_counter) + 1;
g_pixel_mask[thid] = false;
*g_over = true;
}
}
}
}