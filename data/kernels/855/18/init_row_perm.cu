#include "includes.h"
__global__ void init_row_perm(int *d_permutation, int M)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= M) {
return;
}

d_permutation[i] = i;
}