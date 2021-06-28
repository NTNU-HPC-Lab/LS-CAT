#include "includes.h"
__global__ void hello_from_gpu(void)
{
printf("Hello World from the GPU!\n");
}