#include "includes.h"
__global__ void add(int *a, int *b, int *c, int n)
{
//blockDim.x represents threads per block
int index = threadIdx.x + blockIdx.x * blockDim.x;
// as we need to avoid to go beyond the end of the arrays, we need to define the limit
if (index < n)
c[index] = a[index] + b[index];
}