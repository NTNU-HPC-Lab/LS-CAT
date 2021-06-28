#include "includes.h"
__global__ void updatePhi_kernel(int n, bool* d_flags, float* d_energy, float* d_fatigue, float theta) {
unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int stride = blockDim.x * gridDim.x;
while (index < n) {
d_flags[index] = (d_energy[index] - d_fatigue[index]) > theta ? true : false;
index += stride;
}
}