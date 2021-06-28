#include "includes.h"
__global__ void setup_kernel (curandState * state, unsigned long seed )
{
int i= blockDim.x * blockIdx.x + threadIdx.x;

curand_init (seed, i, 0, &state[i]);
}