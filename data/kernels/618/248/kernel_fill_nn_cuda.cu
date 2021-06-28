#include "includes.h"
__global__ void  kernel_fill_nn_cuda(unsigned int *d_nn, int *nearest_neighbour_indexes, unsigned int number_nearest_neighbour_indexes)
{
int ind=blockIdx.x*blockDim.x+threadIdx.x;

if(ind < number_nearest_neighbour_indexes)
{
if(nearest_neighbour_indexes[ind] < 0)
{
d_nn[ind] = 0;
}else
{
d_nn[ind] = 1;
}
}
}