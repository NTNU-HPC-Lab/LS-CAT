#include "includes.h"
__global__ void cudaUpdateActivity_kernel(int * inputs, char * activity, unsigned int * firingRate, unsigned int * exampleFiringRate, int * totalOutput, unsigned long long int * firstEventTime, unsigned long long int * lastEventTime, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ, unsigned int long long timestamp)
{
const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

// One batch per block z dimension
const unsigned int batchInputOffset = blockIdx.z * inputSize;

for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x) {
for (unsigned int y = threadIdx.y; y < inputsDimY; y += blockDim.y) {
for (unsigned int x = threadIdx.x; x < inputsDimX; x += blockDim.x) {

const unsigned int inputsIdx =
x + y*inputsDimX + channel*inputsDimX*inputsDimY;
int act = inputs[inputsIdx + batchInputOffset];
unsigned int actAbs = abs(act);
char spike = act == 0 ? 0 : act/abs(act);

activity[inputsIdx + batchInputOffset] = spike;
firingRate[inputsIdx + batchInputOffset] += actAbs;
exampleFiringRate[inputsIdx + batchInputOffset] += actAbs;
totalOutput[inputsIdx + batchInputOffset] += act;
}
}
}
}