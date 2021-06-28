#include "includes.h"
#define N 1
#define TPB 256


__global__ void mainKernel()
{
printf("Hello world! My threadId is %d\n", threadIdx.x);
}