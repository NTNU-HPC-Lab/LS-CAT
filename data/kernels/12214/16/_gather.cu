#include "includes.h"
__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
const int tid = threadIdx.x + blockDim.x * blockIdx.x;

if (tid < n)
output[tid] = input[(tid % 6) + 6 * indices[tid / 6]];
}