#include "includes.h"
__global__ void var(float * M1, float * M2, float * X, int b, size_t nele) {
int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx<nele) {
float delta = X[idx] - M1[idx];
M1[idx] += delta / (b + 1);
M2[idx] += delta*(X[idx] - M1[idx]);
}
}