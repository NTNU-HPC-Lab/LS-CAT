#include "includes.h"
__global__ void square(int *array, int arrayCount)
{
extern __shared__ int dynamicSmem[];
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if (idx < arrayCount) {
array[idx] *= array[idx];
}
}