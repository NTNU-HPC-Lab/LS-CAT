#include "includes.h"
#ifdef ENABLE_CUDA
#pragma GCC diagnostic push
#pragma GCC diagnostic pop
#endif


#define SIZE 256



__global__ void run_kernel(curandStateMRG32k3a *state, unsigned int *result) {
int id = threadIdx.x + blockIdx.x * SIZE;
curandStateMRG32k3a localState = state[id];
unsigned int x = curand(&localState);
while (x == 0) {
x = curand(&localState);
}
state[id] = localState;
result[id] = x;
}