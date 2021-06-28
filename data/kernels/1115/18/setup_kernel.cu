#include "includes.h"
__global__ void setup_kernel ( curandState * states, unsigned long seed ){
const int tid = threadIdx.x + blockIdx.x * blockDim.x;
curand_init ( seed+tid*4, tid, 0, &states[tid] );
}