#include "includes.h"
__global__ void grayscale(unsigned char * data_rgb, unsigned char * data_gray, std::size_t rows, std::size_t cols)
{
auto i = blockIdx.x * blockDim.x + threadIdx.x;
auto j = blockIdx.y * blockDim.y + threadIdx.y;

if( i < cols && j < rows )
{
data_gray[ j * cols + i ] = (
307 * data_rgb[ 3 * (j * cols + i) ]
+   604 * data_rgb[ 3 * (j * cols + i) + 1 ]
+   113 * data_rgb[ 3 * (j * cols + i) + 2 ]
) / 1024;
}
}