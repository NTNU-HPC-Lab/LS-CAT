#include "includes.h"
__global__ void invert_mass_matrix(double *values, unsigned int size)
{
unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i < size)
{
if (values[i] > 1e-15)
values[i] = 1. / values[i];
else
values[i] = 0.;
}
}