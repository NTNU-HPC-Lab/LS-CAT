#include "includes.h"
__global__ void poisson_rand_kernel(const int64_t seed, int32_t* __restrict numbers, const int64_t maximum, const double lambda) {
extern __shared__ curandState_t curandStateShared[];

int xid = blockIdx.x * blockDim.x + threadIdx.x;
curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
numbers[i] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
}
}