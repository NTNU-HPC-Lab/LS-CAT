#include "includes.h"
__global__ void repeat_x_for_clusters(float * x,int size)
{
int index = blockIdx.x * blockDim.x + threadIdx.x ;
int thread_index = threadIdx.x ;
int block_index = blockIdx.x ;

if (block_index > 0 && index < size)
{
x[index] = x[thread_index] ;
}
}