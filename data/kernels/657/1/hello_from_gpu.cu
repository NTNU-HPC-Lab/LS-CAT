#include "includes.h"
__global__ void hello_from_gpu(void)
{
int bid = blockIdx.x;
int tid = threadIdx.x;
printf("Hello World from block %d and thread %d!\n", bid, tid);
}