#include "includes.h"
__global__ void int_copy(int *vec_to, int *vec_from, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if ( xIndex < n )
vec_to[xIndex] = vec_from[xIndex];

}