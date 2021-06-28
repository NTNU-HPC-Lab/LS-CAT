#include "includes.h"
__global__ void global_max(int *values, int *max, int *reg_maxes, int num_regions, int n)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int region = i % num_regions;
if(i < n)
{
int val = values[i];
if(atomicMax(&reg_maxes[region], val) < val)
{
atomicMax(max, val);
}//end of if statement
}//end of if i < n
}