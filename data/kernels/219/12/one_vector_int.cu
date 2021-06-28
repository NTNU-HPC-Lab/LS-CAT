#include "includes.h"
__global__ void one_vector_int(int *vec, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( xIndex < n )
vec[xIndex]=1;
}