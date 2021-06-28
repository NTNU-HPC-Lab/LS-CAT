#include "includes.h"
__global__ void doGPUWork(int numData, int *data) {
if (threadIdx.x < numData) {
data[threadIdx.x] = threadIdx.x;
}
}