#include "includes.h"
__global__ void setup_kernel(  curandState * state, unsigned long seed )
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
curand_init( seed, id, 0, &state[ id ] );

return;
}