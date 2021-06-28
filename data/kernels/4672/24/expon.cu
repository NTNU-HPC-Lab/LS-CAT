#include "includes.h"
__global__ void expon(float* env, int nhalf) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
if (i < nhalf) {
env[i] = exp(env[i]/nhalf);   // exponentiate
}
}