#include "includes.h"
__global__ void kernel()
{
/*
this just gets some kernel specific parameters
this is just so you can see how non-deterministic thread timing is
*/
int tidx = threadIdx.x + blockIdx.x * blockDim.x;
int tidy = threadIdx.y + blockIdx.y * blockDim.y;

/* print some stuff out */
int size = sizeof(int);
printf("Hello, World! size=%d   tidx=%d, tidy=%d\n", size, tidx, tidy);
return;
}