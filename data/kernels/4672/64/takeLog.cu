#include "includes.h"
__global__ void takeLog(float* input, float* env, int nhalf) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
int j = i<<1;
if (i < nhalf) {
env[i] = log(input[j] > 0.0 ? input[j] : 1e-20);   // take the log of the amplitudes
}
}