#include "includes.h"
__global__ void chooseLaw ( const int nwl, const int *kex, const float *didi11, const float *didi12, const float *didi13, float *didi1 ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
didi1[i] = ( kex[i] == 0 ) * didi11[i] + ( kex[i] == 1 ) * didi12[i] + ( kex[i] == 2 ) * didi13[i];
}
}