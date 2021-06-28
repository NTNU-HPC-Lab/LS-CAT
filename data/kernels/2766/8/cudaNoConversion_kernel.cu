#include "includes.h"
__global__ void cudaNoConversion_kernel(float * data, float * tickOutputsTraces, float * tickOutputsTracesLearning, float scaling, unsigned int inputDimX, unsigned int inputDimY, unsigned int inputDimZ)
{
const unsigned int inputSize = inputDimX * inputDimY * inputDimZ;
const unsigned int batchOffset = blockIdx.x * inputSize;

for (unsigned int idx = threadIdx.x; idx < inputSize; idx += blockDim.x) {
float value = data[idx + batchOffset];
tickOutputsTraces[idx + batchOffset] = scaling*value;
tickOutputsTracesLearning[idx + batchOffset] += scaling*value;
}
}