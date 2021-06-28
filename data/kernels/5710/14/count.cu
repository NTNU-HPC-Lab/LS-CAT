#include "includes.h"
__global__ void count(int *data,int input, int *result)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(data[i] == input)
{
int a = 1;
atomicAdd(result,a);
}
}