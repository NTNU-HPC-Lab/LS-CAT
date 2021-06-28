#include "includes.h"
__global__ void adjacent_difference_simple(int *result, int *input)
{
// compute this thread's global index
unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

if(i > 0)
{
// each thread loads two elements from global memory
int x_i = input[i];
int x_i_minus_one = input[i-1];

// compute the difference using values stored in registers
result[i] = x_i - x_i_minus_one;
}
}