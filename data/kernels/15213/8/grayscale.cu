#include "includes.h"
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
auto i = blockIdx.x * blockDim.x + threadIdx.x;
auto j = blockIdx.y * blockDim.y + threadIdx.y;
if( i < cols && j < rows ) {
g[ j * cols + i ] = (
307 * rgb[ 3 * ( j * cols + i ) ]
+ 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
+ 113 * rgb[  3 * ( j * cols + i ) + 2 ]
) / 1024;
}
}