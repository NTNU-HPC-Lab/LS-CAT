#include "includes.h"
__global__ void axpy(float a, float* x, float* y) {
// RUN: sh -c "test `grep -c -F 'y[hipThreadIdx_x] = a * x[hipThreadIdx_x];' %t` -eq 2"
y[threadIdx.x] = a * x[threadIdx.x];
}