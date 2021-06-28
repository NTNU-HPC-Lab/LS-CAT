#include "includes.h"
__global__ void init_rand(curandState *states, float* randArr) {
unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
curand_init(1234, tid, 0, &states[tid]);  //  Initialize CURAND
randArr[tid] = 2.*M_PI*curand_uniform (&states[tid]);     // between 0 and 1
}