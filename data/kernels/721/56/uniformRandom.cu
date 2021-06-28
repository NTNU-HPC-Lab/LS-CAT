#include "includes.h"
__global__ void uniformRandom( curandState_t *states, float *d_values)
{
int tid = threadIdx.x + blockDim.x * blockIdx.x;
d_values[tid] = curand_uniform(&states[tid]);
}