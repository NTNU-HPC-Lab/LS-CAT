#include "includes.h"
__global__ void copyCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
{
const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

if ( x < width && y < height ) {
out[y * rowSizeOut + x] = in[y * rowSizeIn + x];
}
}