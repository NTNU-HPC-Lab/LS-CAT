#include "includes.h"
__global__ void combineSourceAndBackground ( const int nwl, const int n, const float scale, float *src, const float *bkg ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
if ( i < n && j < nwl ) {
src[i+j*n] = src[i+j*n] + scale * bkg[i+j*n];
}
}