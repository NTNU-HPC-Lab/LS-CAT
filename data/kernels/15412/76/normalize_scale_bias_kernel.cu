#include "includes.h"
__global__ void normalize_scale_bias_kernel(int N, float *x, float *mean, float *variance, float *scales, float *biases, int batch, int filters, int spatial, int inverse_variance, float epsilon)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= N) return;
int f = (index / spatial) % filters;

float val = 0;
if(inverse_variance) val = (x[index] - mean[f]) * variance[f];
else val = (x[index] - mean[f]) / (sqrtf(variance[f] + epsilon));
val *= scales[f];
val += biases[f];

if (!isnan(val) && !isinf(val))
x[index] = val;
}