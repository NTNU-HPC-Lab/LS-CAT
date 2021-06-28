#include "includes.h"
__global__ void mm(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id <= GPUN) {
int i = id / DIM;
int j = id % DIM;
float sum = 0.0f;
for (int k = 0; k < DIM; k++) {
sum += dA[i*DIM+k] * dB[k*DIM+j];
}
dC[id] = sum;
}
}