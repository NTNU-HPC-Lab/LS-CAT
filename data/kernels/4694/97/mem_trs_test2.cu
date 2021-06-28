#include "includes.h"
__global__ void mem_trs_test2(int * input, int size)
{
int gid = blockIdx.x * blockDim.x + threadIdx.x;

if(gid < size)
printf("tid : %d , gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}