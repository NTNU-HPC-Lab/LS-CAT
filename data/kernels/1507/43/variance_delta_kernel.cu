#include "includes.h"
__global__ void  variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= filters) return;
int j,k;
variance_delta[i] = 0;
for(j = 0; j < batch; ++j){
for(k = 0; k < spatial; ++k){
int index = j*filters*spatial + i*spatial + k;
variance_delta[i] += delta[index]*(x[index] - mean[i]);
}
}
variance_delta[i] *= -.5 * powf(variance[i] + .000001f, (float)(-3./2.));
}