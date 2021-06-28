#include "includes.h"
__global__ void arrayOf2DConditions ( const int dim, const int nwl, const float *bn, const float *xx, float *cc ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
cc[t] = ( bn[0+i*2] < xx[t] ) * ( xx[t] < bn[1+i*2] );
}
}