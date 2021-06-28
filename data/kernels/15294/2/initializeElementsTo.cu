#include "includes.h"
__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
a[i] = initialValue;
}