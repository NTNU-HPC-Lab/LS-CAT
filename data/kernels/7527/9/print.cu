#include "includes.h"
__global__ void print()
{
printf("hello from gpu thread %d\n",threadIdx.x);
}