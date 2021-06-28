#include "includes.h"
__global__ void poli4(float* poli, const int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = poli[idx];

if (idx < N)
poli[idx] = 5 + 5 * x + 5 * x * sqrt(x) + 5 * sqrt(x) * x * x + 5 * x *
sqrt(x) * x * x + 5 * x * sqrt(x) * sqrt(x) * x * x;
}