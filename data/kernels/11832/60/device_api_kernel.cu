#include "includes.h"
__global__ void device_api_kernel(curandState *states, float *out, int N)
{
int i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int nthreads = gridDim.x * blockDim.x;
curandState *state = states + tid;

curand_init(9384, tid, 0, state);

for (i = tid; i < N; i += nthreads)
{
float rand = curand_uniform(state);
rand = rand * 2;
out[i] = rand;
}
}