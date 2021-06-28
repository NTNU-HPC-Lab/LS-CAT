#include "includes.h"
__global__ void sum_arrays_gpu(unsigned int * a, unsigned int * b, int size)
{
int index = blockDim.x * blockIdx.x + threadIdx.x;

if (index < size) {
a[0] = a[0] + b[index];
//printf("%u ", a[0]);
}
}