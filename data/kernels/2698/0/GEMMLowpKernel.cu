#include "includes.h"




#define block_count 32
#define thread_per_block 1024
// Wrapper for ATen
__global__ void GEMMLowpKernel(const float* in, const int N, float* out, float scale, float shift, long long qmax, const float* noise, bool enforce_true_zero) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
out[i] = in[i];
if (enforce_true_zero)
out[i] = (out[i] / scale) + shift;
else
out[i] = (out[i] + shift) / scale;
out[i] += noise[i];
out[i] = fminf(out[i], qmax);
out[i] = fmaxf(out[i], 0.);
out[i] = roundf(out[i]);
if (enforce_true_zero)
out[i] = (out[i] - shift) * scale;
else
out[i] = out[i] * scale - shift;
}
}