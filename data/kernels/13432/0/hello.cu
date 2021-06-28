#include "includes.h"

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 16


__global__ void hello()
{
printf("Hello world! I'm a thread %d in block %d\n", threadIdx.x, blockIdx.x);
}