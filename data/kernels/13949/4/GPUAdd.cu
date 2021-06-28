#include "includes.h"
__global__  void GPUAdd(float *array1, float *array2, float *result, int WIDTH)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
result[i] = array1[i] + array2[i];
}