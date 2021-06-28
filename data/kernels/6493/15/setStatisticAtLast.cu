#include "includes.h"
__global__ void setStatisticAtLast ( const int dim, const int nwl, const float *lst, float *stt ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if ( i < nwl ) {
stt[i] = lst[dim+1+i*(dim+1+1+1+1)];
}
}