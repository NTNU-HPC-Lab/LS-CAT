#include "includes.h"
__global__ void hello_kernel(void)
{
// greet from the device : the GPU and its memory
printf("Hello, world from the device!\n");
}