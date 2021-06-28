#include "includes.h"
__global__ void random(double *x, curandState *global_state){
int tid =  blockIdx.x;
curandState local_state;
local_state = global_state[tid];
x[tid] = (double) curand(&local_state);
global_state[tid] = local_state;
}