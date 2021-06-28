#include "includes.h"
__device__ __forceinline__ float relu(float a) {
return a < 0 ? 0 : a;
}
__global__ void relu_derivative(float *upper_grads, float *upper_values, unsigned int upper_size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < upper_size)
if (upper_values[index] == 0)
upper_grads[index] = 0;
}