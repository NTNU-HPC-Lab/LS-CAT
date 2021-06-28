#include "includes.h"
__global__ void AddIntegers(int *arr1, int *arr2, int num_elements)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < num_elements)
{
arr1[id] += arr2[id];
}
}