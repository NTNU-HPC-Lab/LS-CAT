#include "includes.h"
extern "C" {
}
__global__ void u8_to_one_hot_f32(const unsigned char* x, unsigned int nclasses, float* y, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
y[tid*nclasses+x[tid]] = 1.0f;
}
}