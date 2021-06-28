#include "includes.h"
__global__ void cuda_kernel(double *A, double *B, double *C, int arraySize) {
// Get thread ID.
int tid = blockDim.x * blockIdx.x + threadIdx.x;

// Check if thread is within array bounds.
if (tid < arraySize) {
// Add a and b.
C[tid] = A[tid] + B[
tid];
}
}