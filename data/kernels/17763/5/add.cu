#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
// each block handles a different element of the array
// on the device, each block can execute in parallel
// use blockIdx.x to access block index
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}