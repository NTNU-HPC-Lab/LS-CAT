#include "includes.h"
__global__ void saveTheWhalesXX ( const int d0, const int d1, const int i2, float *xxx, const int d3, const int d4, const float *xx ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
if ( i < d3 && j < d4 ) {
xxx[i+j*d0+i2*d0*d1] = xx[i+j*d3];
}
}