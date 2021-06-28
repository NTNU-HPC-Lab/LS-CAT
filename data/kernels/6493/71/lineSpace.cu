#include "includes.h"
__global__ void lineSpace ( const int d, const int n, const float *l, const float *h, float *b ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
float delta;
if ( i < d && j < n ) {
delta = ( h[i] - l[i] ) / ( n - 1 );
b[i+j*d] = l[i] + j * delta;
}
}