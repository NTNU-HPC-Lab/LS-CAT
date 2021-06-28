#include "includes.h"
__global__ void init(int *vector, int N, int val)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;

if (i < N) {
vector[i] = val;
}
}