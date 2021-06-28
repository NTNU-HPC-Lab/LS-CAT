#include "includes.h"


__global__ void polynomial_expansion (float* poly, int degree, int n, float* array) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if( index < n )
{
float out = 0.0;
float xtothepowerof = 1.0;
for ( int x = 0; x <= degree; ++x)
{
out += xtothepowerof * poly[x];
xtothepowerof *= array[index];
}
array[index] = out;
}
}