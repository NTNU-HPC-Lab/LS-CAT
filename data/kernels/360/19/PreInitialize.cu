#include "includes.h"
__global__ void PreInitialize(int size, unsigned int *randoms, int *bestSeen, int *origin, int *mis) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Set the origin to be self
origin[idx] = idx;

// Set MIS to be -1;
mis[idx] = -1;

// Set the bestSeen value to be random
bestSeen[idx] = randoms[idx] % 1000000;
}
}