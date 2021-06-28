#include "includes.h"
__global__ void callOperation(int *niz, int *res, int k, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n) {
return;
}

if (niz[tid] == k) {
atomicAdd(res, 1);
}
}