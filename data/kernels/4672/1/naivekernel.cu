#include "includes.h"
__global__ void naivekernel(float* output, float* frameA, float* frameB, int chans) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = i<<1;
if (i < chans) {
int test = frameA[j] >= frameB[j];
if (test) {
output[j] = frameA[j];
output[j+1] = frameA[j+1];
}
else {
output[j] = frameB[j];
output[j+1] = frameB[j+1];
}
}
}