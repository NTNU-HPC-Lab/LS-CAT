#include "includes.h"
__global__ void chooseDistance ( const int nwl, const int *kex, const float *didi11, float *didi1 ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
didi1[i] = didi11[i+kex[i]*nwl];
}
}