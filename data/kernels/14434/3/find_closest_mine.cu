#include "includes.h"
__global__ void find_closest_mine(float * mine_pos_v, float * distances_v, int * mineIdx_v, int num_sweeprs, int num_mines, float * inputs)
{
#define sweeperIdx blockIdx.y
#define first_item blockIdx.y*num_mines
int my_index = (gridDim.x * blockIdx.x) + threadIdx.x;

//mineIdx_v[sweeperIdx * num_mines + threadIdx.x] = threadIdx.x;
mineIdx_v[sweeperIdx * num_mines + my_index] = my_index;

for (int stride = num_mines / 2; stride > 1; stride /= 2)
{
__syncthreads();
if (my_index < stride)
{
if (distances_v[my_index + first_item] < distances_v[my_index + first_item + stride])
{
distances_v[my_index + first_item] = distances_v[my_index + first_item + stride];
mineIdx_v[my_index + first_item] = mineIdx_v[my_index + first_item + stride];
}
}
}

inputs[sweeperIdx * 4] = mine_pos_v[mineIdx_v[sweeperIdx] * 2];
inputs[sweeperIdx * 4 + 1] = mine_pos_v[mineIdx_v[sweeperIdx] * 2 + 1];

#undef sweeperIdx
#undef first_item
}