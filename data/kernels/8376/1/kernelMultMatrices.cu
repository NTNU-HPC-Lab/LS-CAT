#include "includes.h"
__global__ void kernelMultMatrices(float *a, float *b, float *c,int m, int n) {
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;
//printf("%d,%d\n",i,j);
c[j+i*n]=0;
for(int k=0;k<N;k++) c[j+i*n]+=a[j+k*n]*b[k+i*n];;
__syncthreads();
}