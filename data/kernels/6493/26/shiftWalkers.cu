#include "includes.h"
__global__ void shiftWalkers ( const int dim, const int nwl, const float *xx, const float *x, float *yy ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
yy[t] = xx[t] - x[i];
}
}