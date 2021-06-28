#include "includes.h"
__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY, int size) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
}
}