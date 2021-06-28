#include "includes.h"
__global__ void fixPVandGain(float* input, float* output, float gain, int length) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
int j = i<<1;
if (j < length) {
if (isnan(output[j]))   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
output[j] = 0.0f;  // set to zero any invalid amplitude
if (output[j+1] == -1.0f) {   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
output[j] = 0.0f;   // set to zero the amp related to any undefined frequency
}
else
output[j] *= gain;   // scale all amplitudes by the gain factor
}
if (j == 0) output[0] = input[0];   // keep original DC amplitude
if (j == length-2) output[length-2] = input[length-2];   // keep original Nyquist amplitude
}