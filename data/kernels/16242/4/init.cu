#include "includes.h"
__global__ void init(int* U, int* F, int* d, int startNode, size_t gSize) {
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

if (globalThreadId < gSize) {
U[globalThreadId] = 1;
F[globalThreadId] = 0;
d[globalThreadId] = INT_MAX;
}

if(globalThreadId == 0) {
d[globalThreadId] = 0;
U[globalThreadId] = 0;
F[globalThreadId] = 1;
}
}