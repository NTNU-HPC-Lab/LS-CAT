#include "includes.h"
__device__ int DeviceDefaultStep() {
return gridDim.x * blockDim.x;
}
__device__ int DeviceDefaultIndex() {
return blockIdx.x * blockDim.x + threadIdx.x;
}
__global__ void KernelSelfPlusIters(const bool *indexers, int *iters, int count) {
int index = DeviceDefaultIndex();
int step = DeviceDefaultStep();
for (int i = index; i < count; i += step) {
if (indexers[i]) {
++iters[i];
}
}
}