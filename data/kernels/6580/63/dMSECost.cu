#include "includes.h"
__global__ void dMSECost(float* predictions, float* target, float* dY, int size) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
dY[index] = 2 * (predictions[index] - target[index]);
}
}