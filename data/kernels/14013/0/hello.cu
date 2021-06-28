#include "includes.h"




#define NUM_BLOCKS 16
#define BLOCK_WIDTH 3
// hello world


__global__ void hello()
{
// #if __CUDA_ARCH__ >= 200
printf("Hello world! I'm the %dth thread in %dth block. \n", threadIdx.x, blockIdx.x);
// #endif
}