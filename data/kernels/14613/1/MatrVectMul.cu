#include "includes.h"
__global__ void MatrVectMul(int *d_c, int *d_a, int *d_b)
{
int i = blockIdx.x*blockDim.x+threadIdx.x;
if(i<N)
{
d_c[i]=0;
for (int k=0;k<N;k++)
d_c[i]+=d_a[i+k*N]*d_b[k];
}
}