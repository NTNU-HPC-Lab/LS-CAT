#include "includes.h"
__global__ void setPriorAtLast ( const int dim, const int nwl, const float *lst, float *prr ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
prr[i] = lst[dim+3+i*(dim+1+1+1+1)];
}
}