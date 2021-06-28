#include "includes.h"
__global__ void sum_gradients ( float * gradient, float * new_value )
{
// X Grid iterates all gradient values
int x = blockIdx.x * blockDim.x + threadIdx.x;
// A Simple summation
gradient[x] = __fadd_rz( gradient[x], new_value[x] );
}