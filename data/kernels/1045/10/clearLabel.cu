#include "includes.h"
__global__ void clearLabel(bool *label, unsigned int size)
{
unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id < size)
label[id] = false;
}