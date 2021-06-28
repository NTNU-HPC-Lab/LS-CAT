#include "includes.h"
__global__ void loop()
{
/*
* This idiomatic expression gives each thread
* a unique index within the entire grid.
*/

int i = blockIdx.x * blockDim.x + threadIdx.x;
printf("%d\n", i);
}