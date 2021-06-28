#include "includes.h"
__global__ void mapKex ( const int nwl, const float *r, int *kex ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
kex[i] = ( int ) truncf ( r[i] * ( 3 - 1 + 0.999999 ) );
}
}