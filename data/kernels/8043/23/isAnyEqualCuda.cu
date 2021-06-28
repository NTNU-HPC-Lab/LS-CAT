#include "includes.h"
__global__ void isAnyEqualCuda( const uint8_t * image, uint8_t * value, size_t valueCount, uint32_t width, uint32_t height, uint32_t * differenceCount )
{
const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

if( x < width && y < height )
{
const uint32_t id = y * width + x;

bool equal = false;

for( uint32_t i = 0; i < valueCount; ++i )
{
if( image[id] == value[i] )
{
equal = true;
break;
}
}

if( equal )
atomicAdd( differenceCount, 1 );
}
}