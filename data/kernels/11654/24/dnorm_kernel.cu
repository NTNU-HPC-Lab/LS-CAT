#include "includes.h"
__device__ double dnorm(float x, float mu, float sigma)
{
float std = (x - mu)/sigma;
float e = exp( - 0.5 * std * std);
return(e / ( sigma * sqrt(2 * 3.141592653589793)));
}
__global__ void dnorm_kernel(float *vals, int N, float mu, float sigma)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < N) {
vals[idx] = sigma;
}
}