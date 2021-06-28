#include "includes.h"
__global__ void sgemvn_kernel2_fermi(int n, int m, int n1, float alpha,  float* A, int lda, float *x, float *y)
{
int ind = blockIdx.x*num_threads + threadIdx.x;

A += ind;
x += threadIdx.x;

float res = 0.f;

__shared__ float buff[num_threads];
for(int i=0; i<n1; i += num_threads ){
__syncthreads();
buff[threadIdx.x]  = x[i];

__syncthreads();
#pragma unroll
for(int j=0; j < num_threads ; j++){
res+=A[0]*buff[j];
A+=lda;
}
}
#if 0
__syncthreads();

if (m>n1){
buff[threadIdx.x]  = x[n1];

__syncthreads();
for(int j=0; j<(m-n1); j++){
res += A[0]*buff[j];
A+=lda;
}
}
#endif

if (ind<n)
y[ind] = alpha * res;
}