#include "includes.h"
__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
float partial_cost = target[index] * logf(predictions[index])
+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
atomicAdd(cost, - partial_cost / size);
}
}