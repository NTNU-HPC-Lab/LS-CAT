#include "includes.h"
__global__ void helloFromGPU()
{
if ( threadIdx.x == 0 ) {
printf("Hello World from GPU! %d\n",blockIdx.x);
}
}