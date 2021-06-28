#include "includes.h"
__global__ void print_my_index()
{
int tid = threadIdx.x;
int bid = blockIdx.x;
printf("my id :%d , block_id :%d \n",tid,bid);
}