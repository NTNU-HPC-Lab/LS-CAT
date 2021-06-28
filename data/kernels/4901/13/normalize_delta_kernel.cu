#include "includes.h"
__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;
int f = (index/spatial)%filters;

delta[index] = delta[index] * 1./(sqrt(variance[f]) + .000001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}