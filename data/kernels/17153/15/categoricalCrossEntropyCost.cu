#include "includes.h"
__global__ void categoricalCrossEntropyCost(float *predictions, float *target, int size, float *cost){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size)
{
float partial_cost = target[index] * logf(predictions[index]);
atomicAdd(cost, -partial_cost / size);
}
}