#include "includes.h"
/*
sergeim19
April 27, 2015
Burgers equation - GPU CUDA version
*/


#define NADVANCE (4000)
#define nu (5.0e-2)

__global__ void kernel_calc_uu(double *u_dev, double *uu_dev)
{
int j;
j = blockIdx.x * blockDim.x + threadIdx.x;

uu_dev[j] = 0.5 * u_dev[j] * u_dev[j];
}