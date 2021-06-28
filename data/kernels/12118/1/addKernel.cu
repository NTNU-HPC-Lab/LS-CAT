#include "includes.h"
__global__ void addKernel(int *a, int *b, int *c)
{
// each parallel invocation of add() is referred to as a block.
// The set of blocks is referred to as a grid.
// Each invocation can refer to its block index using blockIdx.x.
// By using blockIdx.x to index into the array, each block handles a different index.
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}