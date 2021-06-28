#include "includes.h"
__global__ void smoothing(float* input, float* output, double alpha, double beta, int length) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
int j = i<<1;
if (j < length) {
output[j] = (float) (input[j] * (1.0 + alpha) - output[j] * alpha);
output[j+1] = (float) (input[j+1] * (1.0 + beta) - output[j+1] * beta);
}
}