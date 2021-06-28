#include "includes.h"
__global__ void compact_1D_array( int * input, int * output, int * output_index_array, int array_size)
{
int gid = blockDim.x*blockIdx.x + threadIdx.x;

//TO DO handle when gid ==0
//this is very unefficient in memory management
if (gid > 0 && gid < array_size)
{
if (output_index_array[gid] != output_index_array[gid - 1])
{
//printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, output_index_array[gid], input[gid], input[gid-1]);
output[output_index_array[gid]] = input[gid-1];
}
}
}