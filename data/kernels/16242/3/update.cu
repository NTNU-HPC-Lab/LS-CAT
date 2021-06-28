#include "includes.h"
__global__ void update(int* U, int* F, int* d, int* del, size_t gSize) {
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

if (globalThreadId < gSize) {
F[globalThreadId] = 0;
if(U[globalThreadId] && d[globalThreadId] < del[0]) {
U[globalThreadId] = 0;
F[globalThreadId] = 1;
}
}
}