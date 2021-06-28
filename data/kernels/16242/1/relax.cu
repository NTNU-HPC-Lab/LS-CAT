#include "includes.h"
__global__ void relax(int* U, int* F, int* d, size_t gSize, int* adjMat) {
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

if (globalThreadId < gSize) {
if (F[globalThreadId]) {
for (int i = 0; i < gSize; i++) {
if(adjMat[globalThreadId*gSize + i] && i != globalThreadId && U[i]) {
atomicMin(&d[i], d[globalThreadId] + adjMat[globalThreadId * gSize + i]);
}
}
}
}
}