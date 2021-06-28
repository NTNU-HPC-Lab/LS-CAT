#include "includes.h"
__global__ void setDistanceAtLast ( const int dim, const int nwl, const float *lst, float *didi ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
didi[i] = lst[dim+i*(dim+1+1+1+1)];
}
}