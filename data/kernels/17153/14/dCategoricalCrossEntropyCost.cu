#include "includes.h"
__global__ void dCategoricalCrossEntropyCost(float *predictions, float *target, float *dY, int size){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size)
{
dY[index] = (predictions[index] - target[index]);
}
}