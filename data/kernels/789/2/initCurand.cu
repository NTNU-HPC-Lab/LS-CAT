#include "includes.h"
__global__ void initCurand(curandState *state, unsigned long seed, int n_rows){
int x = blockDim.x * blockIdx.x + threadIdx.x;
if(x < n_rows) {
curand_init(seed, x, 0, &state[x]);
}
}