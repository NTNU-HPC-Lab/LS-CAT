#include "includes.h"
__global__ void msecost(float* predictions, float* target, int size, float* cost) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
float partial_cost = (predictions[index] - target[index]) * (predictions[index] - target[index]);
atomicAdd(cost,  partial_cost / size);
}
}