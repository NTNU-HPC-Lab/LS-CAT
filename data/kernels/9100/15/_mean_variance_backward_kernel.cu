#include "includes.h"
__global__ void _mean_variance_backward_kernel(float *x, float *grad, float *mean, float *var, int b, int c, int wxh, float *mean_diff, float *var_diff)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x, j, k, ind;

if (i >= c)
return;

mean_diff[i] = 0;
var_diff[i] = 0;
for (j = 0; j < b; ++j) {
for (k = 0; k < wxh; ++k) {
ind = j * c * wxh + i * wxh + k;
mean_diff[i] += grad[ind];
var_diff[i] += grad[ind] * (x[ind] - mean[i]);
}
}
mean_diff[i] *= (-1.0f / sqrt (var[i] + 0.00001f));
var_diff[i] *= -0.5f / (var[i] * sqrtf(var[i]) + 0.00001f);
}