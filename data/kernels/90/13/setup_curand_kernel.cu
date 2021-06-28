#include "includes.h"
__global__ void setup_curand_kernel(curandState *state, int count){
int id = threadIdx.x + blockIdx.x * 64;
if(id < count){
curand_init(1234, id, 0, &state[id]);
}
}