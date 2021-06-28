#include "includes.h"
__global__ void compact2d_1D_array(int * input, int * output, int * output_column_index_array, int * output_row_index_array, int * prev_output_index_array,int* auxiliry_array, int array_size)
{
int gid = blockDim.x*blockIdx.x + threadIdx.x;

//TO DO handle when gid ==0
//this is very unefficient in memory management
if (gid > 0 && gid < array_size)
{
printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, prev_output_index_array[gid], input[gid], input[gid-1]);
if (prev_output_index_array[gid] != prev_output_index_array[gid - 1])
{
//printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, output_index_array[gid], input[gid], input[gid-1]);
output[prev_output_index_array[gid]] = input[gid - 1];
output_column_index_array[prev_output_index_array[gid]] = (gid - 1)% blockDim.x;
}

int colum_index = gid / (blockDim.x  - 1);
int condition = gid % (blockDim.x - 1);

if (condition == 0)
{
printf("column index : %d --- row length : %d \n", condition, prev_output_index_array[gid]);
if (gid == 0)
{
output_row_index_array[0] = 0;
}
else
{
output_row_index_array[colum_index] = prev_output_index_array[gid];
//output_row_index_array[colum_index] = auxiliry_array[colum_index];
}
}
}
}