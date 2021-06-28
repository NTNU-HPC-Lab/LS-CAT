#include "includes.h"
__global__ void setVal( int* testfuck, int size )
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
testfuck[id] = size - id;
}