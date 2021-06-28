#include "includes.h"
__global__ void check_results_kernel( int n, double correctvalue, double * x )
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < n)
{
if ( x[i] != correctvalue )
{
printf("ERROR at index = %d, expected = %f, actual: %f\n",i,correctvalue,x[i]);
}
}
}