#include "includes.h"
__global__ void gpu_helloworld()
{
int threadId = threadIdx.x;
printf("Hello from the GPU! My threadId is %d\n", threadId);
}