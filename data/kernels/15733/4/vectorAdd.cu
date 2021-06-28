#include "includes.h"
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
// Calculate global thread ID (tid)
int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
// Vector boundary guard
if (tid < n) {
// Each thread adds a single element
c[tid] = a[tid] + b[tid];
}
}