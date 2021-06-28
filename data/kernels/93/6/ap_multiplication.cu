#include "includes.h"
__global__ void ap_multiplication(float * values ,int * indeces,float* r ,float * p_sum ,int size)
{
int index = blockDim.x * blockIdx.x + threadIdx.x ;

p_sum[index] = 0;
__syncthreads() ;
if (index < size)
{
for (int i = 0 ; i<3 ; i++)
{
p_sum[index] += values[3*index  + i] * r[indeces[3*index + i]] ;
}
__syncthreads() ;
}
}