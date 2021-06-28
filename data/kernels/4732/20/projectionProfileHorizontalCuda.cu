#include "includes.h"
__global__ void projectionProfileHorizontalCuda( const uint8_t * image, uint32_t rowSize, uint32_t width, uint32_t height, uint32_t * projection )
{
const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

if ( x < width && y < height ) {
const uint8_t * imageX = image + y * rowSize + x;
atomicAdd( &projection[x], (*imageX) );
}
}