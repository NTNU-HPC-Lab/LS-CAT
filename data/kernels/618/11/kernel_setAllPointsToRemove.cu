#include "includes.h"
__global__ void kernel_setAllPointsToRemove(bool *d_markers, int number_of_points)
{
int ind=blockIdx.x*blockDim.x+threadIdx.x;
if(ind<number_of_points)
{
d_markers[ind] = false;
}
}