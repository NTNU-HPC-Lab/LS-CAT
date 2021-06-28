#include "includes.h"

/*
* Refactor `loop` to be a CUDA Kernel. The new kernel should
* only do the work of 1 iteration of the original loop.
*/


__global__ void loop()
{
printf("This is iteration number %d\n", threadIdx.x);
}