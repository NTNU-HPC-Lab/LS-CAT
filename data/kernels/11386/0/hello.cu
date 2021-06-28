#include "includes.h"
__global__ void hello()
{
printf("Hello from Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}