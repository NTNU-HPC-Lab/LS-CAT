#include "includes.h"
__global__ void timeDomainConvolutionNaive(float* ibuf, float* rbuf, float* obuf, long long oframes, long long rframes, int ch, float gain) {
int threadID = blockIdx.x * blockDim.x + threadIdx.x;
float value = 0;
for (int k = 0; k < rframes; k++) {
value += ibuf[threadID - k] * rbuf[k];
}
obuf[threadID * 2 + ch] = value * gain;

}