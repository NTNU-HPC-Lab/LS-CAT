#include "includes.h"
__global__ void reduce(int *a, int *b, int n) {
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
atomicAdd(b, a[i]);
}