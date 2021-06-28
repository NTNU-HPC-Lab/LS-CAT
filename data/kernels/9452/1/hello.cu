#include "includes.h"


// Block index Printf (Rodrigo)


#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1



__global__ void hello()
{
printf("Hello world! I'm thread %d\n", threadIdx.x);
}