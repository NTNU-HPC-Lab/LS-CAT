#include "includes.h"
/*
sergeim19
April 27, 2015
Burgers equation - GPU CUDA version
*/


#define NADVANCE (4000)
#define nu (5.0e-2)

__global__ void kernel_rescale_u(double *u_dev, int N)
{
int j;
j = blockIdx.x * blockDim.x + threadIdx.x;

u_dev[j] = u_dev[j] / (double)N;
}