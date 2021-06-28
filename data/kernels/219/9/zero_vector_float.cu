#include "includes.h"
__global__ void zero_vector_float(float *vec, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( xIndex < n )
vec[xIndex]=0.0f;
}