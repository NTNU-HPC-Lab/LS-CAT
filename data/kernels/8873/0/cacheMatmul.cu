#include "includes.h"

/*
* Read TODO items below
*/




__global__
__global__ void cacheMatmul(float *a, float *b, float *c, int n)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

float acc = 0;
for(int k1=0;k1<n;k1+=gridDim.x)
{
acc=c[i*n+j];
for(int k=k1;k<k1+gridDim.x;k++)
{
acc += a[i*n+k] * b[k*n+j];
}
c[i*n+j] = acc;
}
}