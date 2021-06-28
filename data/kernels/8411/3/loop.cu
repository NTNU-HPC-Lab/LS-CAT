#include "includes.h"
__global__ void loop()
{
printf("This is iteration number %d\n", threadIdx.x + blockIdx.x * blockDim.x);
}