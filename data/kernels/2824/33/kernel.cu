#include "includes.h"
__global__ void kernel(float* polynomial, const size_t N) {
int thread = blockIdx.x * blockDim.x + threadIdx.x;

if (thread < N) {
float x = polynomial[thread];

polynomial[thread] = 3 * x * x - 7 * x + 5;
}
}