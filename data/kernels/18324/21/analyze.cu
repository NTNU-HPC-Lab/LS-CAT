#include "includes.h"
__global__ void analyze(const float *input, float *sum, int numElements) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < numElements) {
atomicAdd(sum + i, input[i]);
}
}