#include "includes.h"
__global__ void sgemvn_kernel1_fermi(int n, int m, int n1, float alpha, float* A, int lda, float *x, float *y)
{
int ind = blockIdx.x*num_threads + threadIdx.x;

A += ind;

float res = 0.f;

for(int i=0; i<n1; i += sgemv_bs ){

#pragma unroll
for(int j=0; j < sgemv_bs ; j++){
res += A[0] * x[j];
A   += lda;
}
x += sgemv_bs;
}

#if 0
if (m>n1){

for(int j=0; j<(m-n1); j++){
res += A[0] * x[j];
A   += lda;
}
}
#endif

if (ind<n)
y[ind] = alpha * res;

}