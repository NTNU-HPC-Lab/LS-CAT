#include "includes.h"
__global__ void Max (const int n, const float *top_temp, float *top_data, float *mask, const int mask_index){
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
if (top_data[index] < top_temp[index])
{
top_data[index] = top_temp[index];
mask[index] = mask_index;
}
}