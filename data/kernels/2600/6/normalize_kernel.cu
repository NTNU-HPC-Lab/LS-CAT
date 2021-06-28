#include "includes.h"

extern "C" {
}


__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;
int f = (index/spatial)%filters;

x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
}