#include "includes.h"
__global__ void generate_uniform_kernel(float *generated_out, curandState_t *state)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

generated_out[idx] = curand_uniform(&state[idx]);
}