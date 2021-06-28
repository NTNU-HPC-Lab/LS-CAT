#include "includes.h"

char* concat(char *s1, char *s2);





__global__ void r_final_sum_and_alpha_calculation(float * r_squared ,float * p_sum ,int size)
{
int index = threadIdx.x ;

__shared__ float shared_r_squared[1024] ;
__shared__ float shared_p_sum[1024] ;

if (index < size)
{
shared_r_squared[index] = r_squared[index]  ;
shared_p_sum[index] = p_sum[index]  ;
} else
{
shared_r_squared[index] = 0 ;
shared_p_sum[index] = 0 ;
}

__syncthreads() ;

for (unsigned int s = blockDim.x/2 ; s> 0 ; s >>= 1)
{
if (index < s)
{
shared_r_squared[index] = shared_r_squared[index] + shared_r_squared[index +s] ;
shared_p_sum[index] = shared_p_sum[index] + shared_p_sum[index +s] ;
__syncthreads() ;
}
}
if(threadIdx.x == 0)
{
//alpha
r_squared[blockIdx.x] = shared_r_squared[0]/shared_p_sum[0] ;

}
}