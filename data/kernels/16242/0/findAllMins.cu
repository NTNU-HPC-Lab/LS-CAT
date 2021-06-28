#include "includes.h"
__global__ void findAllMins(int* adjMat, int* outVec, size_t gSize) {
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
int ind = globalThreadId * gSize;
int min = INT_MAX;

if(globalThreadId < gSize) {
for(int i = 0; i < gSize; i++) {
if(adjMat[ind + i] < min && adjMat[ind + i] > 0) {
min = adjMat[ind + i];
}
}
outVec[globalThreadId] = min;
}
}