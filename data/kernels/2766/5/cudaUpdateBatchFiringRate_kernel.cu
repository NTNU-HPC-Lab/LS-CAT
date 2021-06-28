#include "includes.h"
__global__ void cudaUpdateBatchFiringRate_kernel(unsigned int * firingRate, unsigned int * batchFiringRate, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ, unsigned int batchSize)
{

const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

for (unsigned int channel = blockIdx.x; channel < inputsDimZ; channel += gridDim.x){
for (unsigned int sy = 0; sy < inputsDimY; sy+=blockDim.y){
for (unsigned int sx = 0; sx < inputsDimX; sx+=blockDim.x) {
const unsigned int inputsIdx =
channel*inputsDimX*inputsDimY + sy*inputsDimX + sx;

unsigned int batchSum = 0;
for(unsigned int batch=0; batch<batchSize; ++batch) {
const unsigned int batchInputOffset = batch * inputSize;
batchSum += firingRate[inputsIdx + batchInputOffset];
}
batchFiringRate[inputsIdx] = batchSum;
}
}
}

}