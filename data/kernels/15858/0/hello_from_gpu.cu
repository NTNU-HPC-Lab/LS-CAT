#include "includes.h"


__global__ void hello_from_gpu()
{
printf("Hello World from the GPU!\n");
}