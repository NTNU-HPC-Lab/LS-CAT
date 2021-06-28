#include "includes.h"
__global__ void kernel_curand_init(curandState *state, int seed)
{
// Each possible thread uses same seed, but different sequence number
// (as suggested by CURAND docs)
int global_id = blockDim.x*blockIdx.x + threadIdx.x;
curand_init(seed,global_id,0,&state[global_id]);
}