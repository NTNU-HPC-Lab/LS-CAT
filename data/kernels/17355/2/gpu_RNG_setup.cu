#include "includes.h"

__global__ void gpu_RNG_setup ( curandState * state, unsigned long seed, int N )
{
int id = blockIdx.x * blockDim.x + threadIdx.x;

while(id < N) {

curand_init( (seed << 20) + id, 0, 0, &state[id]);

id += blockDim.x*gridDim.x;
}
}