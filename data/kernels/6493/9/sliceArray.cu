#include "includes.h"
__global__ void sliceArray ( const int n, const int indx, const float *ss, float *zz ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
zz[i] = ss[i+indx];
}
}