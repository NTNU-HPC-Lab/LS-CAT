#include "includes.h"






#define N 100

__global__ void setupRandStates(curandState_t* state, unsigned int seed) {
unsigned block_id = blockIdx.y * gridDim.x + blockIdx.x;
int thread_id = threadIdx.x + block_id * blockDim.x;
// Each thread gets same seed, a different sequence number, no offset
curand_init(seed, thread_id, 0, &state[thread_id]);

}