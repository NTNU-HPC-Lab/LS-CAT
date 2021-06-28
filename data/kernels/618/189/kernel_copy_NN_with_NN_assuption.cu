#include "includes.h"
__global__ void kernel_copy_NN_with_NN_assuption(double *d_temp_double_mem, int *d_nearest_neighbour_indexes, int number_of_points)
{
int index=blockIdx.x*blockDim.x+threadIdx.x;

if(index < number_of_points)
{
int i = d_nearest_neighbour_indexes[index];
if(i != -1)
{
d_temp_double_mem[index] = 1.0f;
}else
{
d_temp_double_mem[index] = 0.0f;
}
}
}