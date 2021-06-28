#include "includes.h"
__global__ void exp_kernel(float* DIST, float pw)
{
register int idx = blockIdx.x * blockDim.x + threadIdx.x;
register float arg = DIST[idx] * pw;
if (arg < -70) arg = -70;
DIST[idx] = exp(arg);
}