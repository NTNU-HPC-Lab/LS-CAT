#include "includes.h"

const float REAL_VALUE_MAX = 1000000.0f;
const int NUM_THREADS = 32;
const int SIZE = 10000;
const int DIMENSION = 2;

__device__ float clamp(float v, float mn = -REAL_VALUE_MAX, float mx = REAL_VALUE_MAX) {
return v < mn ? mn : v > mx ? mx : v;
}
__global__ void updateParticleKernel(float* P, float* V, float* PB, float* GB, float momentum, float introvert, float extrovert, float clamp_min, float clamp_max) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < SIZE * DIMENSION) {
P[i] = clamp(P[i] + V[i], clamp_min, clamp_max);
V[i] = clamp(momentum * V[i] + introvert * (PB[i] - P[i]) + extrovert * (GB[i % DIMENSION] - P[i]), clamp_min, clamp_max);
}
}