#include "includes.h"
__global__ void testKernal()
{
printf("thread number %d\n",threadIdx.x);
}