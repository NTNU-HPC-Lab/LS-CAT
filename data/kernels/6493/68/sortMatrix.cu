#include "includes.h"
__global__ void sortMatrix ( const int nd, const float *a, float *sm ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int ij = i + j * nd;
if ( i < nd && j < nd ) {
sm[ij] = ( a[i] > a[j] );
}
}