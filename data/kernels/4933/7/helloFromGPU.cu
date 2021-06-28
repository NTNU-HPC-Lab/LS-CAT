#include "includes.h"
__global__ void helloFromGPU()
{
printf("Hello from GPU! BlockID: %d - ThreadID: %d.\n", blockIdx.x, threadIdx.x);
}