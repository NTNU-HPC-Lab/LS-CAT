#include "includes.h"
__device__ __forceinline__ float sigmoid(float a) {
return 1.0 / (1.0 + exp (-a));
}
__global__ void sigmoid_derivative(float *upper_grads, float *upper_values, unsigned int upper_size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < upper_size)
upper_grads[index] *= upper_values[index]*(1.0f - upper_values[index]);
}