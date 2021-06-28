#include "includes.h"


// Block index Printf (Rodrigo)


#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1



__global__ void hello()
{
printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}