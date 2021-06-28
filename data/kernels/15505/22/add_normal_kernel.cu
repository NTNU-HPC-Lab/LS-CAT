#include "includes.h"
__global__ void add_normal_kernel(int seed, float *data, int n, float mean, float std) {
if (threadIdx.x != 0) return;
curandState state;

curand_init(seed, 0, 0, &state);
for (size_t i(0); i < n; ++i)
data[i] += curand_normal(&state) * std + mean;
}