#include "includes.h"
__global__ void _norm_backward_kernel(float *x, float *mean, float *var, float *mean_diff, float *var_diff, int b, int c, int wxh, float *grad) {
int ind = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
int j = (ind / wxh) % c;

if (ind >= b * c * wxh) return;

grad[ind] = grad[ind] * 1.0f / (sqrtf(var[j] + 0.00001f)) +
var_diff[j] * 2.0f * (x[ind] - mean[j]) / (wxh * b) +
mean_diff[j] / (wxh * b);
}