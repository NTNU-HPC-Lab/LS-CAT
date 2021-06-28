#include "includes.h"
__global__ void constantArray ( const int n, const float c, float *a ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
a[i] = c;
}
}