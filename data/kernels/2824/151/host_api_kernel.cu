#include "includes.h"
__global__ void host_api_kernel(float *randomValues, float *out, int N)
{
int i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int nthreads = gridDim.x * blockDim.x;

for (i = tid; i < N; i += nthreads)
{
float rand = randomValues[i];
rand = rand * 2;
out[i] = rand;
}
}