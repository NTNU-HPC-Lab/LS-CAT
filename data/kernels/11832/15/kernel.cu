#include "includes.h"
__global__ void kernel(float *F, double *D)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid == 0)
{
*F = 12.1;
*D = 12.1;
}
}