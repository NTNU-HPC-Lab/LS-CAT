#include "includes.h"
__global__ void kernel1(float *dW, float *dWcurr, int N) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < N) {
dWcurr[id] = dW[id];
}
}