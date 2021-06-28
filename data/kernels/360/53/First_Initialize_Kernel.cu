#include "includes.h"
__global__ void First_Initialize_Kernel(int size, unsigned int *randoms, int *bestSeen, int *origin)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Set the origin to be self
origin[idx] = idx;

// Set the bestSeen value to be random
bestSeen[idx] = randoms[idx] % 1000000;
}
}