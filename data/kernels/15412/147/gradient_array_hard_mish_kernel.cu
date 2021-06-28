#include "includes.h"
__device__ float hard_mish_yashas_grad(float x)
{
if (x > 0)
return 1;
if (x > -2)
return x + 1;
return 0;
}
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
__global__ void gradient_array_hard_mish_kernel(int n, float *activation_input_gpu, float *delta)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {

const float x = activation_input_gpu[i];
delta[i] *= hard_mish_yashas_grad(x);
}
}