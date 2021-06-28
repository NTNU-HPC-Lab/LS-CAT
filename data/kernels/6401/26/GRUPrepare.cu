#include "includes.h"
__device__ void finish(unsigned int* counter) {
__syncthreads();
__threadfence();
if (threadIdx.x == 0) { atomicAdd(counter, 1); }
}
__global__ void GRUPrepare(unsigned int* finished, const int round) {
for (int i = 0; i < round; i++) { finished[i] = 0; }
}