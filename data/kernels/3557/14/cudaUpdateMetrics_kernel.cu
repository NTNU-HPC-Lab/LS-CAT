#include "includes.h"
__global__ void cudaUpdateMetrics_kernel(float * inputs, int * activity, long long unsigned int * firingRate, long long unsigned int * totalFiringRate, long long int * outputsActivity, long long int * totalOutputsActivity, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ)
{
const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;
const unsigned int batchInputOffset = blockIdx.z * inputSize;

for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x) {
for (unsigned int y = threadIdx.y; y < inputsDimY; y += blockDim.y) {
for (unsigned int x = threadIdx.x; x < inputsDimX; x += blockDim.x) {

const unsigned int inputsIdx =
x + y*inputsDimX + channel*inputsDimX*inputsDimY;

int value = round(inputs[inputsIdx + batchInputOffset]);
unsigned int event = value == 0 ? 0 : 1;

activity[inputsIdx + batchInputOffset] = event;
firingRate[inputsIdx + batchInputOffset] += event;
totalFiringRate[inputsIdx + batchInputOffset] += event;
outputsActivity[inputsIdx + batchInputOffset] += value;
totalOutputsActivity[inputsIdx + batchInputOffset] += value;
}
}
}
}