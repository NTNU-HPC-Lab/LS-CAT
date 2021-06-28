#include "includes.h"
__global__ void reverse_colors_kernel(int num_rows, int max_color, int *row_colors)
{
int row_id = blockIdx.x * blockDim.x + threadIdx.x;

for ( ; row_id < num_rows ; row_id += blockDim.x * gridDim.x )
{
int color = row_colors[row_id];

if (color > 0)
{
//1 -> max_color
//max_color -> 1
color = max_color - color + 1;
}

row_colors[row_id] = color;
}
}