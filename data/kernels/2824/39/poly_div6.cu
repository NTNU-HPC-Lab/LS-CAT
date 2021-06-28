#include "includes.h"
__global__ void poly_div6(float* poli, const int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < N) {
float x = poli[idx];
float y = 1.0/x;
poli[idx] = 5 + x * ( 7 - x * (9 + x * (5 + x * (5 + x))))+y;
}
}