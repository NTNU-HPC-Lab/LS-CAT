#include "includes.h"

char* concat(char *s1, char *s2);





__global__ void r_initial_sum(float* a , int * indeces , float* x,float * r ,float * r_squared ,float * p_sum ,int size)
{
int index = blockDim.x * blockIdx.x + threadIdx.x ;
int local_index = threadIdx.x ;

__shared__ float shared_r_squared[1024] ;
__shared__ float shared_p_sum[1024] ;

shared_r_squared[local_index] = 0 ;
shared_p_sum[local_index] = 0;
__syncthreads() ;

if (index < size)
{
for (int i = 0 ; i<3 ; i++)
{
shared_p_sum[local_index] += a[3*index  + i] * r[indeces[3*index + i]] ;
}
__syncthreads() ;

shared_r_squared[local_index] = r[index] * r[index] ;
shared_p_sum[local_index] = shared_p_sum[local_index] * r[index] ;
}

__syncthreads() ;
for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
{
if (threadIdx.x < s)
{
shared_r_squared[local_index] = shared_r_squared[local_index] + shared_r_squared[local_index +s] ;
shared_p_sum[local_index] = shared_p_sum[local_index] + shared_p_sum[local_index +s] ;
__syncthreads() ;
}

}

if (threadIdx.x == 0)
{
r_squared[blockIdx.x] = shared_r_squared[0];
p_sum[blockIdx.x] = shared_p_sum[0];
__syncthreads() ;
}
}