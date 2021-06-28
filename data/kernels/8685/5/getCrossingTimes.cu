#include "includes.h"
__global__ void getCrossingTimes(double *results, int *crossTimes, int N, int numSims, int lowerThreshold, int upperThreshold) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < N * numSims) {
if (crossTimes[tid/N] == 0) {
if (results[tid] <= lowerThreshold) {
crossTimes[tid/N] = tid % N;
}
else if (results[tid] >= upperThreshold) {
crossTimes[tid/N] = tid % N;
}
}
tid += blockDim.x + gridDim.x;
}
}