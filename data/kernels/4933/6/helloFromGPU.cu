#include "includes.h"
__global__ void helloFromGPU(void)
{
printf("Hello from GPU - block: %d - thread: %d. \n", blockIdx.x, threadIdx.x);
}