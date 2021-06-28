#include "includes.h"
__global__ void print_threadIds()
{
printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n",
threadIdx.x, threadIdx.y, threadIdx.z);
}