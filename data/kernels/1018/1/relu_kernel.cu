#include "includes.h"
__device__ __forceinline__ float relu(float a) {
return a < 0 ? 0 : a;
}
__global__ void relu_kernel(float *vec, int len) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < len) {
vec[index] = relu(vec[index]);
}
}