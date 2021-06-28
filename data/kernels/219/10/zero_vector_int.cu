#include "includes.h"
__global__ void zero_vector_int(int *vec, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( xIndex < n ){
int z=0;
vec[xIndex]=z;
}
}