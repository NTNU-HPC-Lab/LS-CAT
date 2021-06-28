#include "includes.h"
__global__ void poli_warp(float* poli, const int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;

float x;
if (idx < N) {
x = poli[idx];
poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))- 1.0f/x + 3.0f/(x*x) + x/5.0f;
}
poli[idx] = x;
}