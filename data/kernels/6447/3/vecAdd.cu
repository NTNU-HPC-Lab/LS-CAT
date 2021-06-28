#include "includes.h"
__global__ void vecAdd(float* C, float* A, float* B, int n) {
// Get our global thread ID
int id = blockIdx.x * blockDim.x + threadIdx.x;

// Make sure we do not go out of bounds
if (id < n) {
C[id] = A[id] + B[id];
}
}