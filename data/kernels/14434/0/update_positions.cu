#include "includes.h"

float * g_outputs_d, *g_sweepers_d_2;

__global__ void update_positions(float max_speed, float * outputs_d, float * sweepers_d)
{
int my_index = blockIdx.x * blockDim.x + threadIdx.x;

sweepers_d[my_index] +=  (2 * outputs_d[my_index] * max_speed) - max_speed;
}