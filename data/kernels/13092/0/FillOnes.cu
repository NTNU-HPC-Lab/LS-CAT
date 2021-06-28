#include "includes.h"
__global__ void FillOnes( float *vec, int size ) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if( idx >= size ) {
return;
}

vec[ idx ] = 1.0f;
}