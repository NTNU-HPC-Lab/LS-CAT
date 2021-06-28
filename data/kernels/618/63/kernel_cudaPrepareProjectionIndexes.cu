#include "includes.h"
__global__ void kernel_cudaPrepareProjectionIndexes(char *d_v_is_projection, int  *d_nearest_neighbour_indexes,	int number_of_points)
{
int ind=blockIdx.x*blockDim.x+threadIdx.x;

if(ind<number_of_points)
{
if(d_v_is_projection[ind] == 0)
{
d_nearest_neighbour_indexes[ind] = -1;
}else
{
d_nearest_neighbour_indexes[ind] = ind;
}
}
}