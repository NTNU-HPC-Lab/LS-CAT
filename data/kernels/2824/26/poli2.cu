#include "includes.h"
__global__ void poli2(float* poli, const int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = poli[idx];

if (idx < N)
poli[idx] = 4 * x * x * x + 3 * x * x - 7 * x + 5;
}