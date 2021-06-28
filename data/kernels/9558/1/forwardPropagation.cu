#include "includes.h"






__global__ void forwardPropagation(float *a_d , float *b_d ,int size)
{
int idx = threadIdx.x;
int idy = threadIdx.y;

__shared__ float temp[16][16];

temp[idy][idx] = a_d[(idy * (size+1)) + idx] ;

for(int i = 1 ; i < size; i++)
{
if((idy + i) < size)
{
float var1 =(-1)*(temp[i-1][i-1] / temp[i+idy][i-1]);
temp[i+idy][idx] = temp[i-1][idx] + ((var1) * (temp[i+idy][idx]));
}
__syncthreads();
}

b_d[idy*(size+1) + idx] = temp[idy][idx];
}