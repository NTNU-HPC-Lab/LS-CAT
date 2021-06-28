#include "includes.h"
__global__ void init_data_kernel( int n, double* x)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if ( i < n )
{
x[i] = n - i;
}
}