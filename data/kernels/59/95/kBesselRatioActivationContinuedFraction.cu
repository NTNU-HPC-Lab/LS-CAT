#include "includes.h"
__global__ void kBesselRatioActivationContinuedFraction(float* mat, float* target, float order, int num_terms, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) {
float k = mat[i];
float result = 2 * (order + num_terms) / k;
for(int j = num_terms - 1; j > 0; j--) {
result = 2 * (order + j) / k + 1 / result;
}
target[i] = 1 / result;
}
}