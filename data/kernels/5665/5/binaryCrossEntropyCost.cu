#include "includes.h"
__global__ void binaryCrossEntropyCost(float* cost, float* predictions, float* target, int size) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size) {
float partial_cost = target[index] * logf(1.0e-15+predictions[index])
+ (1.0f - target[index]) * logf(1.0e-15+(1.0f - predictions[index]));
atomicAdd(cost, -1.0 * partial_cost / size);
}
}