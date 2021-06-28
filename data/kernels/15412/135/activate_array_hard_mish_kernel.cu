#include "includes.h"
__device__ float hard_mish_yashas(float x)
{
if (x > 0)
return x;
if (x > -2)
return x * x / 2 + x;
return 0;
}
__device__ float mish_yashas(float x)
{
float e = __expf(x);
if (x <= -18.0f)
return x * e;

float n = e * e + 2 * e;
if (x <= -5.0f)
return x * __fdividef(n, n + 2);

return x - 2 * __fdividef(x, n + 2);
}
__global__ void activate_array_hard_mish_kernel(float *x, int n, float *activation_input, float *output_gpu)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {

float x_val = x[i];
if (activation_input) activation_input[i] = x_val;    // store value before activation
output_gpu[i] = hard_mish_yashas(x_val);
}
}