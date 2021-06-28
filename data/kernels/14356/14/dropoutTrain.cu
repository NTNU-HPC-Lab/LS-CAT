#include "includes.h"
extern "C"

extern "C"
__global__ void dropoutTrain( const float* arguments, float* dropoutMask, float* results, const float dropoutFraction, const long size ) {
const int X = gridDim.x;
const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

if(index < size) {
const float mask = dropoutFraction < dropoutMask[index];
dropoutMask[index] = mask;
results[index] = mask * arguments[index];
}
}