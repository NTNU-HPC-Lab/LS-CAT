#include "includes.h"
__global__ void cudaSNearestNeighborKernel(const float* input, size_t inputSizeX, size_t inputSizeY, float* output, size_t outputSizeX, size_t outputSizeY, size_t nbChannels, size_t batchSize)
{
const size_t inputOffset = (blockIdx.z*blockDim.z + threadIdx.z) * (nbChannels*inputSizeY*inputSizeX);
const size_t outputOffset = (blockIdx.z*blockDim.z + threadIdx.z) * (nbChannels*outputSizeY*outputSizeX);

const float multy = ((float) inputSizeY)/((float) outputSizeY);
const float multx = ((float) inputSizeX)/((float) outputSizeX);

for(size_t channel = blockIdx.x; channel < nbChannels; channel += gridDim.x) {
for(size_t oy = threadIdx.y; oy < outputSizeY; oy += blockDim.y) {
for(size_t ox = threadIdx.x; ox < outputSizeX; ox += blockDim.x) {
const size_t iy = (size_t) oy*multy;
const size_t ix = (size_t) ox*multx;


output[outputOffset +
channel*outputSizeY*outputSizeX +
oy*outputSizeX +
ox] = input[inputOffset +
channel*inputSizeY*inputSizeX +
iy*inputSizeX +
ix];

}
}
}
}