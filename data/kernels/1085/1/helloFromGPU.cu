#include "includes.h"
__global__ void helloFromGPU(void)
{
if (threadIdx.x == 5)
{
printf("Hello World from GPU thread %d!\n", threadIdx.x);
}
}