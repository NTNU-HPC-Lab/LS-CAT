#include "includes.h"
__global__ void chainFunction ( const int dim, const int nwl, const int nst, const int ipr, const float *smpls, float *chnFnctn ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * nwl;
if ( i < nwl && j < nst ) {
chnFnctn[t] = smpls[ipr+t*dim];
}
}