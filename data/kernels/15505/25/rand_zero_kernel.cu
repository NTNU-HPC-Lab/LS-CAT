#include "includes.h"
__global__ void rand_zero_kernel(float *data, int n, float p, curandStatePhilox4_32_10_t *states) {
int x(threadIdx.x + blockDim.x * blockIdx.x);

curandStatePhilox4_32_10_t &state(states[x]);

x *= 4;
float4 vals = curand_uniform4(&state);
for (int i(0); i < 4; ++i, ++x) {
if (x >= n) return;
if (reinterpret_cast<float*>(&vals)[i] < p)
data[x] = 0;
}
}