#include "includes.h"
__global__ void setWalkersAtLast ( const int dim, const int nwl, const float *lst, float *xx ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int t = i + j * dim;
if ( i < dim && j < nwl ) {
xx[t] = lst[i+j*(dim+1+1+1+1)];
}
}