#include "includes.h"
__global__ void init_random(unsigned long long *seed, curandState  *global_state){
int tid = blockIdx.x;
unsigned long long local_seed = seed[tid];
curandState local_state;
local_state = global_state[tid];
curand_init(local_seed,tid,0, &local_state);
global_state[tid] = local_state;
}