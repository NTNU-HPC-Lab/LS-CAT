#include "includes.h"
__global__ void kernel_rand_init(curandState *__restrict__ pState, int seed) {
const int gtid_x = blockIdx.x * blockDim.x + threadIdx.x;
const int gtid_y = blockIdx.y * blockDim.y + threadIdx.y;
const int gtid = gtid_y * gridDim.x * blockDim.x + gtid_x;
curandState state;
curand_init(seed, gtid, 0, &state);
pState[gtid] = state;
}