#include "includes.h"
__global__ void updateWalkers ( const int dim, const int nwl, const float *xx1, const float *q, const float *r, float *xx0 ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
//if ( q[j] > r[j] ) {
xx0[t] = ( q[j] > r[j] ) * xx1[t] + ( q[j] <= r[j] ) * xx0[t];
//}
}
}