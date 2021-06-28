#include "includes.h"
__global__ void BaseNeuronSetFloatArray(float *arr, int n_elem, int step, float val)
{
int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
if (array_idx<n_elem) {
arr[array_idx*step] = val;
}
}