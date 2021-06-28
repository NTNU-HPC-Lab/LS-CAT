#include "includes.h"
__global__ void BaseNeuronSetIntPtArray(int *arr, int *pos, int n_elem, int step, int val)
{
int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
if (array_idx<n_elem) {
arr[pos[array_idx]*step] = val;
}
}