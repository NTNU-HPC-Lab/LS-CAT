#include "includes.h"
__global__ void stream(float *dA, float *dB, float *dC, float alpha, int N) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < N) {
dA[id] = dB[id] + alpha * dC[id];
}
}