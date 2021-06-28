#include "includes.h"
__global__ void refill_randoms(float *dRand, int N, curandState *states)
{
int i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int nthreads = gridDim.x * blockDim.x;
curandState *state = states + tid;

for (i = tid; i < N; i += nthreads)
{
dRand[i] = curand_uniform(state);
}
}