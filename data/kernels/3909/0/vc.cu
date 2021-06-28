#include "includes.h"
__global__ void vc(float *dA, float *dB, int N) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < N) {
dA[id] = dB[id];
}
}