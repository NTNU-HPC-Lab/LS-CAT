#include "includes.h"
__global__ void _norm_forward_kernel(float *x, float *mean, float *variance, int b, int c, int wxh)
{
int ind = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
int j = (ind / wxh) % c;

if (ind >= b * c * wxh)
return;

x[ind] = (x[ind] - mean[j]) / (sqrt(variance[j] + 0.000001f));
}