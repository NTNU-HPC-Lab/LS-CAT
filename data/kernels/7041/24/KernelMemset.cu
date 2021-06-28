#include "includes.h"
__device__ int DeviceDefaultStep() {
return gridDim.x * blockDim.x;
}
__device__ int DeviceDefaultIndex() {
return blockIdx.x * blockDim.x + threadIdx.x;
}
__global__ void KernelMemset(bool *p, int len, bool value) {
int index = DeviceDefaultIndex();
int step = DeviceDefaultStep();
for (int i = index; i < len; i+= step) {
p[i] = value;
}
}