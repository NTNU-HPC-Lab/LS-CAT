#include "includes.h"
__global__ void initializeAtRandom ( const int dim, const int nwl, const float dlt, const float *x0, const float *stn, float *xx ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
xx[t] = x0[i] + dlt * stn[t];
}
}