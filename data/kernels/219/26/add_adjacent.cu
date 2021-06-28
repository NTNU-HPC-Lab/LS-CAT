#include "includes.h"
__global__ void add_adjacent(int *vec, int *vec_shorter, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if ( xIndex < n )
vec_shorter[xIndex] = vec[2 * xIndex] + vec[(2 * xIndex) +1];

}