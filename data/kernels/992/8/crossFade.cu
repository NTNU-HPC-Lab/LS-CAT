#include "includes.h"
__global__ void crossFade(float* out1, float* out2, int numFrames){
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
float fn = float(threadID) / (numFrames - 1.0f);
out1[threadID * 2] = out1[threadID * 2] * (1.0f - fn) + out2[threadID * 2] * fn;
out1[threadID * 2 + 1] = out1[threadID * 2 + 1] * (1.0f - fn) + out2[threadID * 2 + 1] * fn;
}