#include "includes.h"
__global__ void _mean_variance_forward_kernel(float *x, int b, int c, int wxh, float *mean, float *var) {
float scale = 1.0f / (b * wxh);
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x, j,
k, ind;
if (i >= c) return;

mean[i] = 0;
for (j = 0; j < b; ++j) {
for (k = 0; k < wxh; ++k) {
ind = j * c * wxh + i * wxh + k;
mean[i] += x[ind];
var[i] += x[ind] * x[ind];
}
}
mean[i] *= scale;
var[i] = var[i] * scale - mean[i] * mean[i];
}