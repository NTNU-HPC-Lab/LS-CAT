#include "includes.h"
__global__ void SetupPoissKernel(curandState *curand_state, uint64_t n_dir_conn, unsigned long long seed)
{
uint64_t blockId   = (uint64_t)blockIdx.y * gridDim.x + blockIdx.x;
uint64_t i_conn = blockId * blockDim.x + threadIdx.x;
if (i_conn<n_dir_conn) {
curand_init(seed, i_conn, 0, &curand_state[i_conn]);
}
}