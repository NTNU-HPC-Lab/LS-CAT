#include "includes.h"
__global__ void add_normal_kerneld(int seed, double *data, int n, double mean, double std) {
if (threadIdx.x != 0) return;
curandState state;
curand_init(seed, 0, 0, &state);
for (size_t i(0); i < n; ++i)
data[i] += curand_normal_double(&state) * std + mean;
}