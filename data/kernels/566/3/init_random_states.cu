#include "includes.h"
__global__ void init_random_states(unsigned int seed, curandState_t* states, size_t num_states)
{
int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
if(thread_id > num_states)
return;

curand_init(seed, thread_id, 0, &states[thread_id]);
}