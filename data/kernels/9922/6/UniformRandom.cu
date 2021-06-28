#include "includes.h"
__global__ void UniformRandom(double *x, curandState *global_state){
int tid =  blockIdx.x;
curandState local_state;
local_state = global_state[tid];
x[tid] = (double) curand_uniform(&local_state);
global_state[tid] = local_state;
}