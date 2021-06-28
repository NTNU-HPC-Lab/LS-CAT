#include "includes.h"
__global__ void histogramCuda( const uint8_t * data, uint32_t rowSize, uint32_t width, uint32_t height, uint32_t * histogram )
{
const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

if ( x < width && y < height ) {
const uint32_t id = y * rowSize + x;
atomicAdd( &histogram[data[id]], 1 );
}
}