#include "includes.h"
__global__ void sliceIntArray ( const int n, const int indx, const int *ss, int *zz ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < n ) {
zz[i] = ss[i+indx];
}
}