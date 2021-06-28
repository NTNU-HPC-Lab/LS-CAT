#include "includes.h"
__global__ void kernel_push1_start_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight, int *g_sink_weight, int *g_push_reser, int *g_relabel_mask, int *g_graph_height, int *g_height_write, int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish)
{
int x1 = threadIdx.x;
int y1 = threadIdx.y;
int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
int thid = __umul24(y, width1) + x;

__shared__ int height_fn[356];


int temp_mult = __umul24(y1 + 1, 34) + x1 + 1, temp_mult1 = __umul24(y1, 32) + x1;

height_fn[temp_mult] = g_graph_height[thid];

(threadIdx.x == 31 && x < width1 - 1) ? height_fn[temp_mult + 1] = (g_graph_height[thid + 1]) : 0;
(threadIdx.x == 0 && x > 0) ? height_fn[temp_mult - 1] = (g_graph_height[thid - 1]) : 0;
(threadIdx.y == 7 && y < rows1 - 1) ? height_fn[temp_mult + 34] = (g_graph_height[thid + width1]) : 0;
(threadIdx.y == 0 && y > 0) ? height_fn[temp_mult - 34] = (g_graph_height[thid - width1]) : 0;

__syncthreads();

int flow_push = 0, min_flow_pushed = 0;
flow_push = g_push_reser[thid];


if (thid < graph_size1 && g_relabel_mask[thid] == 1 && x < width - 1 && x > 0 && y < rows - 1 && y > 0)
{
int temp_weight = 0;


temp_weight = g_sink_weight[thid];
min_flow_pushed = flow_push;

if (temp_weight > 0 && flow_push > 0 && height_fn[temp_mult] == 1)
{
(temp_weight < flow_push) ? min_flow_pushed = temp_weight : 0;
temp_weight = temp_weight - min_flow_pushed;
g_sink_weight[thid] = temp_weight;
atomicSub(&g_push_reser[thid], min_flow_pushed);

flow_push = flow_push - min_flow_pushed;
}



}


__syncthreads();
min_flow_pushed = g_left_weight[thid];

(flow_push > 0 && (((height_fn[temp_mult] == height_fn[temp_mult - 1] + 1) && min_flow_pushed > 0) || ((height_fn[temp_mult] == height_fn[temp_mult + 1] + 1) && g_right_weight[thid] > 0) || ((height_fn[temp_mult] == height_fn[temp_mult + 34] + 1) && g_down_weight[thid] > 0) || ((height_fn[temp_mult] == height_fn[temp_mult - 34] + 1) && g_up_weight[thid] > 0) || (height_fn[temp_mult] == 1 && g_sink_weight[thid] > 0))) ? g_relabel_mask[thid] = 1 : g_relabel_mask[thid] = 0;



if (thid < graph_size1 && x < width - 1 && x > 0 && y < rows - 1 && y > 0)
{
if (g_sink_weight[thid] > 0)
{
g_height_write[thid] = 1;
}
else
{
int min_height = graph_size;
(min_flow_pushed > 0 && min_height > height_fn[temp_mult - 1]) ? min_height = height_fn[temp_mult - 1] : 0;
(g_right_weight[thid] > 0 && min_height > height_fn[temp_mult + 1]) ? min_height = height_fn[temp_mult + 1] : 0;
(g_down_weight[thid] > 0 && min_height > height_fn[temp_mult + 34]) ? min_height = height_fn[temp_mult + 34] : 0;
(g_up_weight[thid] > 0 && min_height > height_fn[temp_mult - 34]) ? min_height = height_fn[temp_mult - 34] : 0;
g_height_write[thid] = min_height + 1;
}
}

}