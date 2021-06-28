#include "includes.h"
__global__ void generate_kernel(unsigned int *generated_out, curandState_t *state)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

generated_out[idx] = curand(&state[idx]);
}