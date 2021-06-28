#include "includes.h"
__global__ void kBesselRatioActivation(float* mat, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) {
float r = mat[i];
target[i] = cyl_bessel_i1f(r) / cyl_bessel_i0f(r);
}
}