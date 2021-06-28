#include "includes.h"
__global__ void normArray ( const int n, float *a ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
float c = a[0];
if ( i < n ) {
a[i] = a[i] / c;
}
}