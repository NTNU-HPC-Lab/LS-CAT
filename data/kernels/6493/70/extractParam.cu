#include "includes.h"
__global__ void extractParam ( const int d, const int n, const int Indx, const float *s, float *a ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
a[i] = s[Indx+i*d];
}
}