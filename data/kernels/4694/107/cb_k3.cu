#include "includes.h"
__global__ void cb_k3()
{
int gid = blockDim.x * blockIdx.x + threadIdx.x;
if (gid == 0)
{
printf("This is a test 3 \n");
}
}