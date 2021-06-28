#include "includes.h"
__global__ void axpy(float a, float* x, float* y) {
y[threadIdx.x] = a * x[threadIdx.x];
}