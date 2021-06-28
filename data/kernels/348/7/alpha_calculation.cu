#include "includes.h"
__global__ void alpha_calculation(float * r_squared ,float * p_sum,float* alpha)
{
alpha[0] = r_squared[0]/p_sum[0] ;
}