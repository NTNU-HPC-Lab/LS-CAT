#include "includes.h"
__device__ __forceinline__ float sigmoid(float a) {
return 1.0 / (1.0 + exp (-a));
}
__global__ void sigmoid_kernel(float *vec, int len) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < len) {
vec[index] = sigmoid(vec[index]);
}
}