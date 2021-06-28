#include "includes.h"
__global__ void minimumCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
{
const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

if ( x < width && y < height ) {
const uint8_t * in1X = in1 + y * rowSizeIn1 + x;
const uint8_t * in2X = in2 + y * rowSizeIn2 + x;
uint8_t * outX = out + y * rowSizeOut + x;
(*outX) = ((*in1X) < (*in2X)) ? (*in1X) : (*in2X);
}
}