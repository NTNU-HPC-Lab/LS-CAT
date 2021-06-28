#include "includes.h"
__global__ void fmad_kernel(double x, double y, double *out)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid == 0)
{
*out = x * x + y;
}
}