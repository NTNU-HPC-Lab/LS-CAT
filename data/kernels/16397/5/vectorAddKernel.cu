#include "includes.h"
__global__ void vectorAddKernel(float* A, float* B, float* Result) {
// insert operation here
int i = threadIdx.x + blockDim.x * blockIdx.x;
Result[i] = A[i] + B[i];
}