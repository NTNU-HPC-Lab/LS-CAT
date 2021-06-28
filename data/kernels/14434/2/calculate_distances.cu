#include "includes.h"
__global__ void calculate_distances(float * sweeper_pos_v, float * mine_pos_v, int num_sweepers, int num_mines, float * distance_v, float * inputs, int * sweeper_score_v, int width, int height, int size)
{
#define sweeperIdx blockIdx.y
#define mineIdx threadIdx.x*2

int distanceIdx = (blockIdx.y * num_mines) + threadIdx.x;
float vec_x;
float vec_y;
float distance;

__shared__ float sweeper_pos[2];

if (threadIdx.x < 2)
{
sweeper_pos[threadIdx.x] = sweeper_pos_v[sweeperIdx + threadIdx.x];
inputs[((sweeperIdx * 4) + threadIdx.x) + 2] = sweeper_pos[threadIdx.x]; //copy the sweeper position out to the inputs for the neural network in parallel

}

__syncthreads();


vec_x = mine_pos_v[mineIdx] - sweeper_pos[0];
vec_y = mine_pos_v[mineIdx + 1] - sweeper_pos[1];
distance = sqrt((vec_x * vec_x) + (vec_y * vec_y));
distance_v[distanceIdx] = distance;

if (distance < size)
{
/*
mine_pos_v[mineIdx] = width / 2;
mine_pos_v[mineIdx + 1] = height / 2;
*/

mine_pos_v[mineIdx] = ((threadIdx.x + 1 ) * clock()) % width;
mine_pos_v[mineIdx + 1] = ((threadIdx.x + 1) * clock()) % height;


sweeper_score_v[sweeperIdx]++;
}

#undef sweeperIdx
#undef mineIdx
}