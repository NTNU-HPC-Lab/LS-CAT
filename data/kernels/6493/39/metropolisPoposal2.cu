#include "includes.h"
__global__ void metropolisPoposal2 ( const int dim, const int nwl, const int isb, const float *xx, const float *rr, float *xx1 ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
xx1[t] = xx[t] + ( i == isb ) * rr[j];
}
}