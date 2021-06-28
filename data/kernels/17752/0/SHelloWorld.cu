#include "includes.h"
__global__ void SHelloWorld()
{
printf("SubHelloWorld from %d-%d\n", blockIdx.x, threadIdx.x);
}