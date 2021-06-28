#include "includes.h"
__global__ void cudaSReduceIndex_kernel(  const unsigned int inputSize, const unsigned int inputBatchOffset, const unsigned int outputBatchOffset, const float* valueThreshold, const float* inputs, int* outputMap, float* scores)
{
const int batchPos = blockIdx.z;
const int clsPos = blockIdx.y;

const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

const int inputIndex = index
+ inputSize*clsPos
+ batchPos*inputBatchOffset;

const int outputIndex = index
+ inputSize*clsPos
+ batchPos*outputBatchOffset;

if(index < inputSize)
{

float value = inputs[inputIndex];

if(value >= valueThreshold[clsPos])
{
outputMap[outputIndex] = index;
scores[outputIndex] = value;
}
else
{
outputMap[outputIndex] = -1;
scores[outputIndex] = -1.0;
}

}
}