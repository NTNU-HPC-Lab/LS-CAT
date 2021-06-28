#include "includes.h"
__global__ void randoms(curandState_t* states, float* numbers, float lower, float higher) {
int index = blockDim.x * blockIdx.x + threadIdx.x;
numbers[index] = lower + (higher - lower) * curand_uniform(&states[index]);
}