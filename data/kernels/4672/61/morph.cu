#include "includes.h"
__global__ void morph(float* output, float* input1, float* input2, float ampCoeff, float freqCoeff, int length) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = i<<1;
if (j  < length) {
output[j] = input1[j]*(1.0-ampCoeff) + input2[j]*(ampCoeff);
output[j+1] = input1[j+1]*(1.0-freqCoeff) + input2[j+1]*(freqCoeff);
}
}