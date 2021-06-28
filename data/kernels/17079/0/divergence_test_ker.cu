#include "includes.h"
__global__ void divergence_test_ker()
{
if(threadIdx.x % 2 == 0)
printf("threadIdx.x %d : This is an even thread.\n", threadIdx.x);
else
printf("threadIdx.x %d : This is an odd thread.\n", threadIdx.x);
}